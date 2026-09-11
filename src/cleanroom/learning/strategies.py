"""The bandit's action space, and the context features that condition it.

Two ideas live here:

* **Strategies** are the discrete arms the agent chooses between. Each one is a
  genuinely different way to turn a web page into rows, so which one wins is a
  real empirical question rather than a cosmetic label.
* **Buckets** are the context. A pricing table and a press release need different
  strategies, so a single global ranking would average away the signal. Bucketing
  the page first is what makes this a *contextual* bandit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    id: str
    summary: str
    #: Appended to the extractor-synthesis prompt. This is the only place the
    #: chosen arm influences the generated code, which keeps the causal link
    #: between "arm pulled" and "reward observed" clean.
    prompt_hint: str


STRATEGIES: dict[str, Strategy] = {
    "table_parse": Strategy(
        id="table_parse",
        summary="Parse markdown/HTML tables structurally, one row per table row.",
        prompt_hint=(
            "Locate the pipe-delimited markdown tables or <table> elements. Map the "
            "header cells onto the schema fields and emit one row per table row. Use "
            "only string splitting and regex -- do not attempt semantic inference."
        ),
    ),
    "heading_sections": Strategy(
        id="heading_sections",
        summary="Split on headings, treat each section as one record.",
        prompt_hint=(
            "Split the document on markdown headings (lines starting with '#') or on "
            "blank-line-separated blocks. Treat each section as exactly one record and "
            "pull each schema field out of that section's text with targeted regex."
        ),
    ),
    "regex_fields": Strategy(
        id="regex_fields",
        summary="One tuned regex per schema field, scanned over the whole page.",
        prompt_hint=(
            "Write one specific regex per schema field and scan the entire document "
            "with each. Zip the per-field match lists together positionally into rows. "
            "Prefer anchored patterns that include the surrounding label text."
        ),
    ),
    "label_value_pairs": Strategy(
        id="label_value_pairs",
        summary="Detect 'Label: value' and definition-list shapes.",
        prompt_hint=(
            "Detect label/value pairs -- 'Field: value', 'Field - value', definition "
            "lists, and two-column layouts. Group consecutive pairs into records, "
            "starting a new record whenever a label you have already filled repeats."
        ),
    ),
    "list_items": Strategy(
        id="list_items",
        summary="One record per bullet/numbered list item.",
        prompt_hint=(
            "Treat each bullet or numbered list item as one record. Within an item, "
            "split on common delimiters (comma, en dash, pipe, parentheses) and assign "
            "the resulting fragments to schema fields by position and by type sniffing."
        ),
    ),
}

STRATEGY_IDS: tuple[str, ...] = tuple(STRATEGIES)

BUCKETS: tuple[str, ...] = ("table_heavy", "list_heavy", "sectioned", "prose")


# -- context featurisation ---------------------------------------------------

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_HTML_TABLE = re.compile(r"<t[rd]\b", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+\S", re.MULTILINE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)


@dataclass(frozen=True)
class PageFeatures:
    """Cheap, deterministic shape features. No model call, so it is free."""

    table_rows: int
    list_items: int
    headings: int
    lines: int

    @property
    def bucket(self) -> str:
        # Deliberately ordered: structural signals beat textual ones, because a
        # page with a table and some prose is still best handled as a table.
        if self.table_rows >= 3:
            return "table_heavy"
        if self.list_items >= 5 and self.list_items > self.headings:
            return "list_heavy"
        if self.headings >= 3:
            return "sectioned"
        return "prose"


def featurise(document: str) -> PageFeatures:
    text = document or ""
    return PageFeatures(
        table_rows=len(_TABLE_ROW.findall(text)) + len(_HTML_TABLE.findall(text)) // 2,
        list_items=len(_LIST_ITEM.findall(text)),
        headings=len(_HEADING.findall(text)),
        lines=text.count("\n") + 1,
    )


def bucket_for(document: str) -> str:
    """Map a fetched page to one of `BUCKETS`."""
    return featurise(document).bucket
