"""Offline Wikipedia (Simple English ZIM) — the environment's library.

World-side only.  Used to derive stage-1 questions and, from stage 3, as the
organism's `look up` action.  See docs/design/incubator_matron.md §4.
"""
from __future__ import annotations
import html as _html
import os
import re
from libzim.reader import Archive
from libzim.search import Query, Searcher

ZIM = os.environ.get(
    "WIKI_ZIM", os.path.expanduser("~/data/wikipedia/wikipedia_en-simple_all_nopic_2026-09.zim"))
_DROP = re.compile(r"<(script|style|table|sup|figure)[^>]*>.*?</\1>", re.S)
_P = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_TAG = re.compile(r"<[^>]+>")


class Wiki:
    def __init__(self, path: str = ZIM):
        self.z = Archive(path)

    def __len__(self) -> int:
        return self.z.entry_count

    def paragraphs(self, entry) -> list[str]:
        """Plain-text paragraphs of an article entry (tables, refs dropped)."""
        item = entry.get_item()
        if not item.mimetype.startswith("text/html"):
            return []
        page = _DROP.sub(" ", bytes(item.content).decode("utf-8", "ignore"))
        out = []
        for p in _P.findall(page):
            t = re.sub(r"\s+", " ", _html.unescape(_TAG.sub("", p))).strip()
            t = re.sub(r"\s+([.,;:)])", r"\1", t)
            if len(t) > 40:
                out.append(t)
        return out

    def entry(self, i: int):
        e = self.z._get_entry_by_id(i)
        return None if e.is_redirect else e

    def search(self, query: str, n: int = 3) -> list[str]:
        return list(Searcher(self.z).search(Query().set_query(query)).getResults(0, n))

    def by_path(self, path: str):
        return self.z.get_entry_by_path(path)
