#!/usr/bin/env python3
"""
Fetch quotes from ldsquotations.com topic pages (e.g. https://ldsquotations.com/topic/jesus-christ/)
and export them as an InkyPi-style card JSON payload.

This uses the WordPress REST API (wp-json) rather than scraping paginated HTML.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Iterable
from urllib.parse import urlparse

import requests


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in {"br", "p", "div", "blockquote", "li"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "blockquote", "li"}:
            self._parts.append("\n")

    def text(self) -> str:
        raw = unescape("".join(self._parts))
        lines = [line.strip() for line in raw.splitlines()]
        lines = [line for line in lines if line]
        return "\n\n".join(lines)


def _html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html or "")
    return parser.text()


def _topic_slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = (parsed.path or "/").strip("/")
    match = re.search(r"(?:^|/)topic/([^/]+)(?:/|$)", path)
    if not match:
        raise ValueError(f"Could not determine topic slug from URL path: {parsed.path!r}")
    return match.group(1).strip()


def _wp_api_base(base_url: str) -> str:
    return base_url.rstrip("/") + "/wp-json/wp/v2"


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    timeout: float = 30.0,
) -> requests.Response:
    resp = session.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "InkyPi-QuoteFetcher/1.0"},
    )
    resp.raise_for_status()
    return resp


@dataclass(frozen=True)
class TagSpec:
    id: int
    name: str
    slug: str


def _find_tag_by_slug(session: requests.Session, api_base: str, slug: str) -> TagSpec:
    resp = _request_json(session, f"{api_base}/tags", params={"slug": slug, "per_page": 100})
    matches = resp.json() or []
    if not matches:
        raise ValueError(f"No tag found for slug={slug!r}")
    tag = matches[0]
    return TagSpec(id=int(tag["id"]), name=str(tag.get("name") or slug), slug=str(tag.get("slug") or slug))


def _speaker_from_embedded_terms(post: dict) -> str:
    embedded = post.get("_embedded") or {}
    terms = embedded.get("wp:term") or []
    for group in terms:
        if not group:
            continue
        taxonomy = group[0].get("taxonomy")
        if taxonomy != "category":
            continue
        # ldsquotations.com appears to use categories for speakers/authors.
        names = [str(term.get("name") or "").strip() for term in group]
        names = [name for name in names if name]
        return names[0] if names else ""
    return ""


def _iter_posts_for_tag(
    session: requests.Session,
    api_base: str,
    *,
    tag_id: int,
    per_page: int,
    max_pages: int | None,
    sleep_seconds: float,
) -> Iterable[dict]:
    per_page = max(1, min(100, int(per_page)))
    page = 1
    seen = 0
    total_pages = None

    while True:
        if max_pages is not None and page > max_pages:
            return
        resp = _request_json(
            session,
            f"{api_base}/posts",
            params={"tags": tag_id, "per_page": per_page, "page": page, "_embed": 1},
        )
        if total_pages is None:
            try:
                total_pages = int(resp.headers.get("X-WP-TotalPages") or "0") or None
            except ValueError:
                total_pages = None

        posts = resp.json() or []
        if not posts:
            return
        for post in posts:
            yield post
            seen += 1

        if total_pages is not None and page >= total_pages:
            return
        page += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Export LDSQuotations topic quotes to JSON")
    parser.add_argument("--topic-url", help="Topic page URL, e.g. https://ldsquotations.com/topic/jesus-christ/")
    parser.add_argument("--topic-slug", help="Topic slug, e.g. jesus-christ")
    parser.add_argument("--base-url", default="https://ldsquotations.com", help="Base site URL")
    parser.add_argument("--out", help="Write JSON output to this path (default: stdout)")
    parser.add_argument("--id", dest="card_id", help="Override output id (default: slug with underscores)")
    parser.add_argument("--label", help="Override output label")
    parser.add_argument("--type", default="quote", help="Output type field (default: quote)")
    parser.add_argument("--per-page", type=int, default=100, help="Posts per API page (max 100)")
    parser.add_argument("--max-pages", type=int, help="Limit API pages fetched (debugging)")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds to sleep between API pages")
    args = parser.parse_args(argv)

    if not args.topic_slug and not args.topic_url:
        parser.error("Provide --topic-slug or --topic-url")

    slug = (args.topic_slug or "").strip()
    if not slug:
        slug = _topic_slug_from_url(args.topic_url or "")

    api_base = _wp_api_base(args.base_url)
    with requests.Session() as session:
        tag = _find_tag_by_slug(session, api_base, slug)
        items: list[dict] = []
        seen_texts: set[str] = set()

        for post in _iter_posts_for_tag(
            session,
            api_base,
            tag_id=tag.id,
            per_page=args.per_page,
            max_pages=args.max_pages,
            sleep_seconds=max(0.0, float(args.sleep)),
        ):
            text = _html_to_text((post.get("content") or {}).get("rendered") or "")
            if not text:
                continue
            key = re.sub(r"\s+", " ", text.strip())
            if key in seen_texts:
                continue
            seen_texts.add(key)
            speaker = _speaker_from_embedded_terms(post)
            items.append({"text": text, "attribution": speaker})

    out = {
        "version": 1,
        "id": (args.card_id or slug.replace("-", "_")).strip(),
        "label": (args.label or f"LDS Quotations: {tag.name}").strip(),
        "type": (args.type or "quote").strip(),
        "items": items,
    }

    payload = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

