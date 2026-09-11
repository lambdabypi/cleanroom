"""The loop.

One episode:

    You.com page -> featurise -> bucket
                 -> memory.search(bucket, host)        [what went wrong before]
                 -> bandit.select(bucket)              [which strategy to try]
                 -> Claude writes extract()            [conditioned on the arm]
                 -> Daytona runs it                    [the environment]
                 -> validator scores the rows          [the reward]
                 -> repair once if it failed           [self-repair]
                 -> bandit.update(bucket, arm, reward) [credit assignment]
                 -> memory.add(lesson)                 [textual learning]
                 -> dataset.merge(valid rows)          [the artifact]

Both learning channels run because they fail differently. The bandit generalises
numerically across pages of the same shape but cannot represent "this site hides
the price in a data attribute"; memory carries exactly that but cannot rank
strategies. Together the demo shows a curve going up *and* the agent citing its
own past lesson.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from cleanroom.config import Settings, settings
from cleanroom.learning.bandit import ThompsonBandit
from cleanroom.learning.budget import Profile, ProfileBandit
from cleanroom.learning.memory import Lesson, MemoryStore, lesson_from_failure
from cleanroom.learning.reward import RewardReport, compute_reward
from cleanroom.learning.store import (
    EpisodeRecord,
    episode_count,
    load_bandit,
    load_profile_bandit,
    log_episode,
    save_bandit,
    save_profile_bandit,
    utcnow,
)
from cleanroom.observability.ledger import get_ledger
from cleanroom.learning.strategies import bucket_for, featurise
from cleanroom.partners.daytona_env import RunResult, build_environment
from cleanroom.partners.one_client import OneClient, OneError, PublishResult
from cleanroom.partners.you_client import Source, YouClient
from cleanroom.pipeline.dataset import (
    Dataset,
    clean_data_manifest,
    manifest_markdown,
    record_provenance,
)
from cleanroom.pipeline.schema import load_schema
from cleanroom.pipeline.synthesize import (
    CreditsExhausted,
    Extractor,
    PayloadTooLarge,
    SynthesisError,
    Synthesizer,
    build_synthesizer,
)

#: Below this, an attempt is worth one repair turn. Above it, the strategy is
#: basically working and another Claude call is not worth the latency.
REPAIR_THRESHOLD = 0.55


@dataclass
class EpisodeOutcome:
    episode: int
    source: Source
    bucket: str
    strategy: str
    reward: RewardReport
    report: dict | None
    rows: list[dict]
    repairs: int
    first_reward: float
    lessons_used: list[Lesson]
    lesson_written: Lesson | None
    duration_s: float
    crashed: bool
    extractor: Extractor | None = None
    #: Set when the code writer failed. Such an episode measures the provider,
    #: not the policy, so it is excluded from learning statistics.
    synthesis_error: str | None = None
    repair_error: str | None = None
    #: Execution profile chosen by the second bandit, and what it cost.
    profile: str = "standard"
    utility: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    llm_calls: int = 0

    @property
    def score(self) -> float:
        return self.reward.total

    @property
    def counts_toward_learning(self) -> bool:
        return self.synthesis_error is None


@dataclass
class RunSummary:
    episodes: list[EpisodeOutcome] = field(default_factory=list)
    rows_added: int = 0
    dataset_rows: int = 0
    publish: PublishResult | None = None
    backend: str = ""

    @property
    def scored(self) -> list[EpisodeOutcome]:
        """Episodes that actually measured the policy.

        Provider failures are excluded on purpose: a 402 from the code writer
        scores 0.0 but says nothing about the strategy, and averaging those in
        would make a dead API key look like a broken agent.
        """
        return [e for e in self.episodes if e.counts_toward_learning]

    @property
    def provider_failures(self) -> list[EpisodeOutcome]:
        return [e for e in self.episodes if not e.counts_toward_learning]

    @property
    def mean_reward(self) -> float:
        scored = self.scored
        return sum(e.score for e in scored) / len(scored) if scored else 0.0

    def improvement(self) -> tuple[float, float]:
        """Mean reward over the first third vs the last third of the scored run."""
        scored = self.scored
        n = len(scored)
        if n < 6:
            return (self.mean_reward, self.mean_reward)
        cut = max(2, n // 3)
        early = sum(e.score for e in scored[:cut]) / cut
        late = sum(e.score for e in scored[-cut:]) / cut
        return (early, late)


def _host(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url or "")
    return re.sub(r"^www\.", "", match.group(1).lower()) if match else (url or "")


class Learner:
    """Owns the bandit, the memory, and the partner clients for a run."""

    def __init__(
        self,
        schema: dict | None = None,
        cfg: Settings | None = None,
        *,
        you: YouClient | None = None,
        synthesizer: Synthesizer | None = None,
        environment: Any | None = None,
        memory: MemoryStore | None = None,
        bandit: ThompsonBandit | None = None,
        profiles: ProfileBandit | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg or settings
        self.cfg.ensure_state_dir()
        self.schema = schema or load_schema()
        self.you = you or YouClient(self.cfg)
        self.synth = synthesizer or build_synthesizer(self.cfg)
        self.env = environment or build_environment(self.cfg)
        self.memory = memory or MemoryStore(self.cfg)
        self.bandit = bandit or load_bandit(self.cfg, seed=seed)
        # Second, independent learner: how expensively to solve each page shape.
        self.profiles = profiles or load_profile_bandit(self.cfg, seed=seed)
        self.ledger = get_ledger(self.cfg)
        self.dataset = Dataset(self.schema, self.cfg)
        self._episode_offset = episode_count(self.cfg)

    # -- source acquisition -----------------------------------------------

    def gather_sources(self, topic: str | None = None, *, count: int | None = None) -> list[Source]:
        search_cfg = self.schema.get("search") or {}
        query = topic or search_cfg.get("topic") or self.schema.get("description") or self.schema["name"]
        return self.you.find_sources(
            query,
            count=count or int(search_cfg.get("count") or 8),
            freshness=search_cfg.get("freshness"),
            include_domains=search_cfg.get("include_domains"),
        )

    # -- tool-failure adaptation -------------------------------------------

    #: Smallest document budget worth trying. Below this there is not enough
    #: page left for any strategy to find records.
    MIN_DOC_CHARS = 2_500

    def _synthesize_within_limits(
        self,
        *,
        source: Source,
        strategy: str,
        bucket: str,
        lessons: Sequence[Lesson],
        doc_chars: int,
    ) -> Extractor:
        """Synthesize, halving the document budget if the provider says 413.

        Free-tier providers cap tokens per request, and the profile bandit does
        not know where that cap is. Rather than lose the episode, the agent backs
        off until the request fits -- and records a lesson so the profile bandit
        sees the cost of an oversized budget on the next pass.
        """
        budget = doc_chars
        while True:
            try:
                return self.synth.synthesize(
                    schema=self.schema,
                    document=source.markdown,
                    source_url=source.url,
                    strategy_id=strategy,
                    bucket=bucket,
                    lessons=lessons,
                    doc_chars=budget,
                )
            except PayloadTooLarge:
                if budget <= self.MIN_DOC_CHARS:
                    raise
                budget //= 2
                self.memory.add(
                    f"Provider rejected a {doc_chars}-char document budget as too "
                    f"large; {budget} chars fit. Prefer leaner profiles on this tier.",
                    tags=(bucket, "payload-limit", "provider"),
                    weight=1.4,
                )

    # -- one episode -------------------------------------------------------

    def run_episode(
        self,
        source: Source,
        *,
        episode: int,
        human: int | None = None,
        max_repairs: int = 1,
        greedy: bool = False,
    ) -> EpisodeOutcome:
        started = time.monotonic()
        self.ledger.episode = episode
        cost_before = self.ledger.total_usd
        features = featurise(source.markdown)
        bucket = features.bucket
        expected = self.schema.get("expected_rows_per_page")

        # 1. Recall before acting. The query mixes host and shape so lessons
        #    transfer both site-wise and layout-wise.
        lessons = self.memory.search(f"{_host(source.url)} {bucket} extraction", limit=3)

        # 2. Choose an arm, and separately choose how much to spend on it. The
        #    caller's `max_repairs` is an upper bound the profile cannot exceed.
        strategy = self.bandit.select(bucket, greedy=greedy)
        profile = self.profiles.select(bucket, greedy=greedy)
        repair_budget = min(max_repairs, profile.max_repairs)
        llm_calls = 0

        # Captured before the strategy posterior is updated. Two bandits learning
        # at once confound each other: while the strategy dimension is still
        # exploring, every profile scores badly, the cheapest one wins on cost
        # alone, and the profile posterior can commit to "lean" before the
        # evidence means anything. So the cost dimension only learns from
        # episodes that used the currently-best-known strategy -- on-policy
        # credit assignment for spend.
        #
        # This buys convergence *speed*, not asymptotic correctness: measured
        # over 10 seeds, gating takes the right answer from 7/10 to 10/10 at 30
        # episodes, while both reach 10/10 by 100. Demo runs live in exactly that
        # 20-40 episode window. See tests/test_observability.py.
        best_known_strategy = self.bandit.best_arm(bucket)

        # 3. Write code, run it, score it -- repairing at most `max_repairs` times.
        extractor: Extractor | None = None
        run: RunResult | None = None
        reward: RewardReport = compute_reward(None, crashed=True)
        first_reward = 0.0
        repairs = 0
        repair_error: str | None = None

        try:
            llm_calls += 1
            extractor = self._synthesize_within_limits(
                source=source,
                strategy=strategy,
                bucket=bucket,
                lessons=lessons,
                doc_chars=profile.doc_chars,
            )
        except CreditsExhausted:
            # Never swallow this: every later episode would score 0.0 and the run
            # would look like a broken policy instead of a dead credit balance.
            raise
        except SynthesisError as exc:
            # The code writer failed, which says nothing about the strategy. So
            # the bandit is deliberately NOT updated -- punishing an arm for
            # provider flakiness would corrupt the posterior. But the episode is
            # still logged and flagged, because an unlogged episode makes the
            # run table and the episode log disagree.
            duration = time.monotonic() - started
            log_episode(
                EpisodeRecord(
                    episode=episode,
                    timestamp=utcnow(),
                    url=source.url,
                    bucket=bucket,
                    strategy=strategy,
                    reward=0.0,
                    rows_total=0,
                    rows_valid=0,
                    crashed=True,
                    reward_detail={
                        "stage": "synthesis_failed",
                        "error": str(exc)[:400],
                        "counts_toward_learning": False,
                        "repairs": 0,
                    },
                    memory_hits=[l.text[:160] for l in lessons],
                    duration_s=duration,
                ),
                self.cfg,
            )
            return EpisodeOutcome(
                episode=episode,
                source=source,
                bucket=bucket,
                strategy=strategy,
                reward=reward,
                report=None,
                rows=[],
                repairs=0,
                first_reward=0.0,
                lessons_used=list(lessons),
                lesson_written=None,
                duration_s=duration,
                crashed=True,
                synthesis_error=str(exc),
            )

        while True:
            run = self.env.run(extractor.code, source.markdown, self.schema, source.url)
            reward = compute_reward(
                run.report,
                human=human,
                crashed=run.crashed,
                expected_rows=expected,
            )
            if repairs == 0:
                first_reward = reward.total

            if reward.total >= REPAIR_THRESHOLD or repairs >= repair_budget:
                break

            try:
                llm_calls += 1
                extractor = self.synth.repair(
                    schema=self.schema,
                    document=source.markdown,
                    source_url=source.url,
                    strategy_id=strategy,
                    bucket=bucket,
                    previous=extractor,
                    failure=run.stderr or "produced too few valid rows",
                    report=run.report,
                    lessons=lessons,
                    doc_chars=profile.doc_chars,
                )
            except CreditsExhausted:
                raise
            except SynthesisError as exc:
                # Record why the repair could not happen. Without this a failed
                # repair is indistinguishable from "no repair was needed", which
                # is how a broken repair path hides in plain sight.
                repair_error = str(exc)
                break
            repairs += 1

        # 4. Credit assignment, to both learners.
        #    The strategy bandit sees raw quality; the profile bandit sees quality
        #    minus what the episode cost. Feeding both the same number would make
        #    the profile bandit blind to the only thing it controls.
        self.bandit.update(bucket, strategy, reward.total)
        input_tokens = extractor.input_tokens if extractor else 0
        if not input_tokens:
            # No usage block from the provider: fall back to the budget we asked
            # for, so cost accounting degrades rather than silently reading zero.
            input_tokens = min(len(source.markdown), profile.doc_chars) // 4
        cost_learned = strategy == best_known_strategy
        if cost_learned:
            utility = self.profiles.update(
                bucket,
                profile.id,
                value=reward.total,
                input_tokens=input_tokens * max(1, llm_calls),
                llm_calls=llm_calls,
            )
        else:
            # Still report what the utility would have been, for the log.
            from cleanroom.learning.budget import utility as utility_of

            utility = utility_of(
                value=reward.total,
                input_tokens=input_tokens * max(1, llm_calls),
                llm_calls=llm_calls,
            )

        # 5. Write a lesson, but only when there is something specific to say.
        lesson = lesson_from_failure(
            url=source.url,
            bucket=bucket,
            strategy=strategy,
            reward=reward.total,
            report=run.report if run else None,
            stderr=run.stderr if run else "",
        )
        if lesson:
            self.memory.add(lesson.text, tags=lesson.tags, weight=lesson.weight)

        # 6. Keep the rows.
        rows = list((run.report or {}).get("rows") or []) if run else []
        merged = self.dataset.merge(rows)
        record_provenance(
            source=source.as_provenance(),
            episode=episode,
            strategy=strategy,
            bucket=bucket,
            reward=reward.total,
            rows_contributed=merged.added,
            cfg=self.cfg,
        )

        duration = time.monotonic() - started
        episode_cost = self.ledger.total_usd - cost_before
        outcome = EpisodeOutcome(
            episode=episode,
            source=source,
            bucket=bucket,
            strategy=strategy,
            reward=reward,
            report=run.report if run else None,
            rows=rows,
            repairs=repairs,
            first_reward=first_reward,
            lessons_used=list(lessons),
            lesson_written=lesson,
            duration_s=duration,
            crashed=bool(run.crashed) if run else True,
            extractor=extractor,
            repair_error=repair_error,
            profile=profile.id,
            utility=utility,
            cost_usd=episode_cost,
            input_tokens=input_tokens,
            llm_calls=llm_calls,
        )

        # Persist after every episode, not just at the end of the run. Writes are
        # atomic (temp file + rename), so a concurrent `cleanroom report` either
        # sees the previous posterior or the new one, never a torn file. Saving
        # only at the end meant a crash at episode 23 threw away 23 episodes of
        # learning, and made `report` useless while a run was in flight.
        save_bandit(self.bandit, self.cfg)
        save_profile_bandit(self.profiles, self.cfg)
        # The dataset is the run's actual artifact and it was also only written
        # at the end -- interrupting a run published an empty CSV because every
        # extracted row was still in memory.
        self.dataset.save()

        log_episode(
            EpisodeRecord(
                episode=episode,
                timestamp=utcnow(),
                url=source.url,
                bucket=bucket,
                strategy=strategy,
                reward=reward.total,
                rows_total=int((run.report or {}).get("rows_total") or 0) if run else 0,
                rows_valid=int((run.report or {}).get("rows_valid") or 0) if run else 0,
                crashed=outcome.crashed,
                reward_detail={
                    **reward.as_dict(),
                    "stage": (run.report or {}).get("stage", "unknown") if run else "unknown",
                    "counts_toward_learning": True,
                    "repairs": repairs,
                    "repair_error": repair_error,
                    "first_reward": round(first_reward, 4),
                    "backend": extractor.backend if extractor else "",
                    "cache_read_tokens": extractor.cache_read_tokens if extractor else 0,
                    "rows_added": merged.added,
                    # Efficiency half of the loop.
                    "profile": profile.id,
                    "doc_chars_budget": profile.doc_chars,
                    "input_tokens": input_tokens,
                    "llm_calls": llm_calls,
                    "utility": round(utility, 4),
                    "cost_learned": cost_learned,
                    "cost_usd": round(episode_cost, 6),
                    # Why rows failed. Their absence is what hid a bug where
                    # every row was rejected for a missing source_url -- the log
                    # showed reward 0.0 with no reason attached.
                    "field_error_counts": (run.report or {}).get("field_error_counts") if run else {},
                    "sample_errors": ((run.report or {}).get("sample_errors") or [])[:5] if run else [],
                },
                memory_hits=[l.text[:160] for l in lessons],
                duration_s=duration,
            ),
            self.cfg,
        )
        return outcome

    # -- full run ----------------------------------------------------------

    def run(
        self,
        *,
        episodes: int = 12,
        topic: str | None = None,
        publish: bool = False,
        greedy: bool = False,
        max_repairs: int = 1,
        on_episode=None,
    ) -> RunSummary:
        summary = RunSummary(backend=getattr(self.env, "backend", "unknown"))

        sources = self.gather_sources(topic)
        if not sources:
            raise RuntimeError(
                "You.com returned no pages with usable content. Try a broader topic, "
                "or relax 'freshness' in the schema's search block."
            )

        for index in range(episodes):
            # Cycle the source list when there are more episodes than pages.
            # Revisiting a page is legitimate here: a different arm gets sampled,
            # so the posterior still gains information.
            source = sources[index % len(sources)]
            outcome = self.run_episode(
                source,
                episode=self._episode_offset + index + 1,
                max_repairs=max_repairs,
                greedy=greedy,
            )
            summary.episodes.append(outcome)
            if on_episode:
                on_episode(outcome)

        save_bandit(self.bandit, self.cfg)
        save_profile_bandit(self.profiles, self.cfg)
        self.dataset.save()
        summary.rows_added = sum(
            int((e.report or {}).get("rows_valid") or 0) for e in summary.episodes
        )
        summary.dataset_rows = len(self.dataset.rows)

        if publish:
            summary.publish = self.publish(dry_run=False)

        return summary

    # -- the loop-closing write -------------------------------------------

    def publish(self, *, dry_run: bool = True) -> PublishResult:
        """Push the dataset and its provenance manifest out through One."""
        one = OneClient(self.cfg)
        manifest = clean_data_manifest(self.schema, self.cfg)
        name = self.schema.get("name") or "dataset"

        try:
            result = one.publish_dataset(
                filename=f"data/{name}.csv",
                content=self.dataset.to_csv(),
                message=(
                    f"cleanroom: {len(self.dataset.rows)} rows of {name} "
                    f"from {manifest.get('distinct_sources')} sources"
                ),
                dry_run=dry_run,
            )
        except OneError as exc:
            return PublishResult(False, str(exc))

        if result.ok and not dry_run:
            try:
                one.publish_dataset(
                    filename=f"data/{name}.PROVENANCE.md",
                    content=manifest_markdown(manifest),
                    message=f"cleanroom: provenance for {name}",
                    dry_run=False,
                )
            except OneError:
                # The dataset landed; a missing manifest is not worth failing on.
                pass
        return result

    def close(self) -> None:
        closer = getattr(self.env, "close", None)
        if closer:
            closer()

    def __enter__(self) -> "Learner":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
