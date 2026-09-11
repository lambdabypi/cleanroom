"""One client -- credential layer and, more importantly, the write-back.

One is what closes the loop. The judging criterion listed first is "did the agent
go past producing an answer and change something in a real system?", and here
that means the cleaned dataset lands in an actual GitHub repo (or Slack, or
Notion) through One's managed connection, with the call visible in One's
execution log at app.withone.ai/logs.

One exposes a deliberately small surface -- four tools, discovered by search
rather than enumerated:

    list_one_integrations -> search_one_platform_actions
                          -> get_one_action_knowledge -> execute_one_action

Discovery goes through the `one` CLI; execution goes through the passthrough API
with the three required headers. Two of One's documented house rules are enforced
in code rather than left to the model: **read the action knowledge before
executing** (`execute` refuses an unvetted action id unless explicitly
overridden), and **confirm before any write** (`dry_run` defaults to on).

On CLI output shapes: `one`'s `--agent` flag emits JSON, but subcommand flags do
move between releases. Every parse here degrades to returning raw text instead of
raising, and setting `ONE_PUBLISH_ACTION_ID` skips discovery altogether -- that is
the path to use if the CLI fights you during the event.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Sequence
from urllib.parse import quote

import requests

from cleanroom.config import Settings, settings

#: The passthrough mirrors the target API's own path: One's CLI resolves
#: `/repos/{owner}/{repo}/contents/{path}` to
#: `https://api.withone.ai/v1/passthrough/repos/.../contents/...` and uses the
#: action's own HTTP method. Verified by inspecting the `request.url` One returns
#: on a successful execute.
PASSTHROUGH_BASE = "https://api.withone.ai/v1/passthrough"
DEFAULT_TIMEOUT = 90

#: Windows routes npm's `one` shim through cmd.exe, which truncates command
#: lines at 8191 characters. A base64 CSV blows past that ("The command line is
#: too long"), so anything this size or larger goes over HTTP instead of the CLI.
CLI_ARG_BUDGET = 6000


class OneError(RuntimeError):
    pass


#: Keys One nests real payloads under. Observed on CLI 1.56.1: a successful
#: execute returns `{"dryRun": false, "request": {...}, "response": {...}}`, so
#: the platform's own fields sit one level down under `response`. Reading the
#: top level only silently finds nothing, which for the sha lookup means every
#: update is attempted as a create and fails.
_ENVELOPE_KEYS = ("response", "data", "result", "body")


def unwrap_field(payload: Any, field: str) -> str | None:
    """Pull `field` out of a One response, whatever envelope it arrived in."""
    if isinstance(payload, dict):
        if payload.get(field):
            return str(payload[field])
        for key in _ENVELOPE_KEYS:
            found = unwrap_field(payload.get(key), field)
            if found:
                return found
    return None


@dataclass
class Action:
    action_id: str
    name: str = ""
    platform: str = ""
    description: str = ""
    #: HTTP verb and URL template of the underlying API call. Needed to reach the
    #: action over the passthrough rather than the CLI.
    method: str = ""
    path: str = ""
    raw: dict = field(default_factory=dict)

    def resolve_path(self, path_vars: dict | None = None) -> str:
        """Substitute `{{var}}` / `{var}` placeholders in the path template.

        Values are percent-encoded with **no safe characters**, so a `/` inside a
        variable becomes `%2F`. This is load-bearing: One's passthrough matches
        routes by path segment, so a literal slash in a variable adds a segment,
        the route stops matching, and the upstream API answers 404. Measured
        against the GitHub contents endpoint:

            contents/probe.txt           -> 201
            contents/data/probe.txt      -> 404   (extra segment)
            contents/data%2Fprobe.txt    -> 201   (stored as data/probe.txt)
        """
        resolved = self.path
        for key, value in (path_vars or {}).items():
            encoded = quote(str(value), safe="")
            for pattern in (f"{{{{{key}}}}}", f"{{{key}}}"):
                resolved = resolved.replace(pattern, encoded)
        return resolved


@dataclass
class PublishResult:
    ok: bool
    detail: str
    action_id: str | None = None
    dry_run: bool = False
    response: Any = None


class OneClient:
    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        self._cli = shutil.which("one")
        self._knowledge_read: set[str] = set()

    # -- capability reporting ---------------------------------------------

    @property
    def has_cli(self) -> bool:
        return bool(self._cli)

    @property
    def has_secret(self) -> bool:
        return bool(self.cfg.one_secret)

    def status(self) -> dict[str, Any]:
        return {
            "cli_installed": self.has_cli,
            "secret_set": self.has_secret,
            "connection_keys_set": bool(self.cfg.one_connection_keys),
            "publish_platform": self.cfg.one_publish_platform,
            "publish_target": self.cfg.one_publish_target or None,
            "pinned_action_id": os.getenv("ONE_PUBLISH_ACTION_ID") or None,
        }

    # -- CLI plumbing ------------------------------------------------------

    def _cli_json(self, args: Sequence[str], *, timeout: int = 45) -> Any:
        if not self._cli:
            raise OneError("the `one` CLI is not installed (npm i -g @withone/cli)")
        # `--agent` is a global flag and must precede the subcommand.
        cmd = [self._cli, "--agent", *args]
        try:
            # encoding must be explicit: `text=True` alone decodes with the
            # locale codepage, and on Windows (cp1252) One's UTF-8 output -- it
            # uses typographic apostrophes in action titles -- raises
            # UnicodeDecodeError inside subprocess's reader thread.
            done = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OneError(f"`one {' '.join(args)}` failed to run: {exc}") from exc

        if done.returncode != 0:
            raise OneError(
                f"`one {' '.join(args)}` exited {done.returncode}: "
                f"{(done.stderr or done.stdout or '').strip()[:400]}"
            )
        out = (done.stdout or "").strip()
        if not out:
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            # Tolerated on purpose: a CLI that prints a table instead of JSON
            # should not take the run down.
            return {"_raw": out}

    def list_integrations(self) -> Any:
        """Step 1 of the four-tool loop: what is actually connected."""
        return self._cli_json(["list"])

    def search_actions(self, query: str, *, platform: str, limit: int = 5) -> list[Action]:
        """Step 2: find actions by plain-language query, not a giant tool list.

        `one actions search <platform> <query>` -- platform is a positional, not
        a flag. Verified against One CLI 1.56.1.
        """
        payload = self._cli_json(
            ["actions", "search", platform, query, "--type", "execute"]
        )

        rows: list[dict] = []
        if isinstance(payload, list):
            rows = [r for r in payload if isinstance(r, dict)]
        elif isinstance(payload, dict):
            for key in ("actions", "results", "data", "items"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    rows = [r for r in candidate if isinstance(r, dict)]
                    break

        actions: list[Action] = []
        for row in rows[:limit]:
            action_id = row.get("actionId") or row.get("action_id") or row.get("id")
            if not action_id:
                continue
            actions.append(
                Action(
                    action_id=str(action_id),
                    name=str(row.get("name") or row.get("title") or ""),
                    platform=str(row.get("platform") or platform or ""),
                    description=str(row.get("description") or "")[:400],
                    method=str(row.get("method") or "").upper(),
                    path=str(row.get("path") or ""),
                    raw=row,
                )
            )
        return actions

    def get_action_knowledge(self, action_id: str, *, platform: str) -> Any:
        """Step 3: the action's schema is authoritative -- read it before executing.

        `one actions knowledge <platform> <actionId>`; the platform positional is
        required.
        """
        knowledge = self._cli_json(["actions", "knowledge", platform, action_id])
        self._knowledge_read.add(action_id)
        return knowledge

    # -- execution ---------------------------------------------------------

    def execute(
        self,
        action_id: str,
        body: dict | None = None,
        *,
        platform: str,
        path_vars: dict | None = None,
        query_params: dict | None = None,
        connection_key: str | None = None,
        require_knowledge: bool = True,
        action: Action | None = None,
    ) -> Any:
        """Step 4: run the action.

        **Path variables go in `--path-vars`, not the body.** One's own action
        knowledge is blunt about this: "Do NOT pass path variables or query
        parameters in the -d body flag -- this causes 403 errors." A GitHub
        file write has `owner`, `repo` and `path` in the URL template, so putting
        them in the body produces a 403 that looks like a permissions problem
        and is actually a shape problem.
        """
        if require_knowledge and action_id not in self._knowledge_read:
            raise OneError(
                f"refusing to execute {action_id} before its knowledge was read "
                "(One's house rule: the schema is authoritative). Call "
                "get_action_knowledge() first, or pass require_knowledge=False."
            )

        key = connection_key or self.cfg.one_connection_keys
        if not key:
            raise OneError(
                "ONE_CONNECTION_KEYS is not set. Run `one add github`, then "
                "`one list` and copy the key (it looks like "
                "live::github::default::abc123)."
            )

        payload = json.dumps(body) if body else ""
        too_big = len(payload) >= CLI_ARG_BUDGET

        # Prefer HTTP whenever the action's method and path are known. The CLI's
        # execute path is the weaker option on two counts: Windows truncates its
        # command line at 8191 bytes, and `one actions execute` has been observed
        # aborting with a libuv assertion (exit 0xC0000409) on this platform.
        # Discovery (search/knowledge/list) still goes through the CLI -- those
        # payloads are small and the CLI is the documented interface for them.
        if action and action.method and action.path and self.has_secret:
            return self._execute_passthrough(
                action_id,
                body or {},
                connection_key=key,
                method=action.method,
                path=action.resolve_path(path_vars),
                query_params=query_params,
            )

        if self.has_cli:
            if too_big:
                raise OneError(
                    f"payload is {len(payload)} bytes, over the {CLI_ARG_BUDGET}-byte "
                    "CLI limit, and no action method/path was supplied to use the "
                    "HTTP passthrough instead. Pass the Action from search_actions()."
                )
            args = ["actions", "execute", platform, action_id, key]
            if body:
                args += ["-d", payload]
            if path_vars:
                args += ["--path-vars", json.dumps(path_vars)]
            if query_params:
                args += ["--query-params", json.dumps(query_params)]
            return self._cli_json(args, timeout=120)

        raise OneError(
            "no `one` CLI and no action method/path available; cannot execute "
            f"{action_id}"
        )

    def _execute_passthrough(
        self,
        action_id: str,
        body: dict,
        *,
        method: str,
        path: str,
        connection_key: str | None = None,
        query_params: dict | None = None,
    ) -> Any:
        """Execute over HTTP instead of the CLI.

        The passthrough URL mirrors the target API's path and uses the action's
        own HTTP verb, with the connection and action identified by headers.
        Unlike the CLI this has no argument-length ceiling, which is what makes
        publishing a real dataset possible on Windows.
        """
        if not self.has_secret:
            raise OneError("ONE_SECRET is not set; run `cleanroom doctor`")

        key = connection_key or self.cfg.one_connection_keys
        if not key:
            raise OneError("ONE_CONNECTION_KEYS is not set; `one add <platform>` first")

        url = f"{PASSTHROUGH_BASE}/{path.lstrip('/')}"
        headers = {
            "x-one-secret": self.cfg.one_secret,
            "x-one-connection-key": key,
            "x-one-action-id": action_id,
            "Content-Type": "application/json",
        }
        try:
            resp = requests.request(
                method.upper(),
                url,
                headers=headers,
                params=query_params or None,
                json=body if body else None,
                timeout=DEFAULT_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise OneError(f"One passthrough unreachable: {exc}") from exc

        if resp.status_code in (401, 403):
            raise OneError(
                f"One rejected the credentials ({resp.status_code}) for {method} {path}. "
                "Re-check ONE_SECRET and ONE_CONNECTION_KEYS."
            )
        if not resp.ok:
            # The upstream platform's own error arrives here; it is far more
            # useful than a generic message, so pass it through verbatim.
            raise OneError(f"{method} {path} -> {resp.status_code}: {resp.text[:400]}")

        # One returns application/octet-stream even for JSON, so do not branch on
        # the content type.
        try:
            return resp.json()
        except ValueError:
            return {"_raw": resp.text[:2000]}

    # -- the loop-closing call --------------------------------------------

    def publish_dataset(
        self,
        *,
        filename: str,
        content: str,
        message: str,
        dry_run: bool = True,
    ) -> PublishResult:
        """Write the cleaned dataset to the configured platform.

        `dry_run` defaults to True. Publishing is an outward-facing write to a
        real repo, so it happens only when a caller asks for it explicitly -- the
        CLI does that via `cleanroom run --publish`.
        """
        target = self.cfg.one_publish_target
        platform = (self.cfg.one_publish_platform or "github").lower()
        if not target:
            return PublishResult(False, "ONE_PUBLISH_TARGET is not set; nothing published")

        if dry_run:
            return PublishResult(
                True,
                f"[dry run] would write {filename} ({len(content)} bytes) to {platform}:{target}",
                dry_run=True,
            )

        found = self.search_actions(
            "create or update a file in a repository", platform=platform
        )
        pinned = os.getenv("ONE_PUBLISH_ACTION_ID")
        write_action = next(
            (a for a in found if a.action_id == pinned),
            found[0] if found else None,
        )
        if write_action is None:
            return PublishResult(
                False,
                f"no {platform} file-write action found via One; check "
                f'`one actions search {platform} "create or update file"`',
            )
        action_id = write_action.action_id

        # House rule: read the schema before executing against it.
        try:
            self.get_action_knowledge(action_id, platform=platform)
        except OneError as exc:
            return PublishResult(False, f"could not read action knowledge: {exc}", action_id)

        owner, _, repo = target.partition("/")
        if not owner or not repo:
            return PublishResult(
                False,
                f"ONE_PUBLISH_TARGET must be 'owner/repo' (got {target!r})",
                action_id,
            )

        # Only pin a branch when explicitly asked. An empty repository has no
        # branches at all, so naming one 404s; omitting it lets the platform use
        # (and create) its default.
        branch = os.getenv("ONE_PUBLISH_BRANCH") or None

        # owner/repo/path are URL template placeholders. Putting them in the body
        # yields a 403 that reads like a permissions failure.
        path_vars = {"owner": owner, "repo": repo, "path": filename}
        body = {
            "message": message,
            # GitHub's contents API takes base64. Raw text is a 422 whose message
            # ("content is not valid Base64") is at least honest about it.
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        }
        if branch:
            body["branch"] = branch

        # Overwriting an existing file requires its current blob sha, and a demo
        # re-run is exactly that case. GitHub's error ('"sha" wasn't supplied')
        # is clear but arrives too late to be useful, so look it up first.
        sha = self.get_file_sha(
            owner=owner, repo=repo, path=filename, branch=branch, platform=platform
        )
        if sha:
            body["sha"] = sha

        try:
            response = self.execute(
                action_id, body, platform=platform, path_vars=path_vars,
                action=write_action,
            )
        except OneError as exc:
            return PublishResult(False, str(exc), action_id)

        verb = "updated" if sha else "created"
        where = f"{platform}:{target}" + (f"@{branch}" if branch else "")
        return PublishResult(True, f"{verb} {filename} on {where}", action_id, response=response)

    def get_file_sha(
        self,
        *,
        owner: str,
        repo: str,
        path: str,
        branch: str | None = None,
        platform: str = "github",
    ) -> str | None:
        """Current blob sha of a file, or None if it does not exist yet.

        Reads the **git tree** rather than the contents endpoint, because the
        contents endpoint cannot be reached for a nested path through One:
        a literal `/` in the `{{path}}` variable breaks One's segment-based route
        matching (404), and percent-encoding it works for `PUT` but *not* for
        `GET` on GitHub's side. The tree endpoint takes a single-segment ref and
        returns every path with its sha, so it sidesteps the problem and handles
        arbitrary nesting in one call.

        Best-effort: a missing file is the normal first-publish case and must not
        be treated as an error.
        """
        found = self.search_actions("get a repository git tree", platform=platform)
        tree_action = next(
            (a for a in found if a.method == "GET" and "git/trees" in a.path), None
        )
        if tree_action is None:
            return None

        try:
            self.get_action_knowledge(tree_action.action_id, platform=platform)
        except OneError:
            return None

        # Try the requested branch, then the usual defaults -- the ref has to
        # exist or the tree lookup 404s.
        refs = [r for r in (branch, "main", "master") if r]
        for ref in dict.fromkeys(refs):
            try:
                response = self.execute(
                    tree_action.action_id,
                    None,
                    platform=platform,
                    path_vars={"owner": owner, "repo": repo, "treeSha": ref},
                    query_params={"recursive": "1"},
                    action=tree_action,
                )
            except OneError:
                continue

            entries = response.get("tree") if isinstance(response, dict) else None
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if (
                    isinstance(entry, dict)
                    and entry.get("type") == "blob"
                    and entry.get("path") == path
                ):
                    return str(entry.get("sha")) or None
            # Tree read fine and the file simply is not there yet.
            return None
        return None
