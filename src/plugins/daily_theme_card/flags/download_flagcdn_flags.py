#!/usr/bin/env python3
"""
Download flag assets from Flagpedia's CDN (flagcdn.com) and generate a manifest.

Source data:
  - https://flagcdn.com/en/codes.json (code -> English name)

Assets:
  - SVG: https://flagcdn.com/<code>.svg
  - PNG: https://flagcdn.com/w<width>/<code>.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


LOGGER = logging.getLogger(__name__)

CODES_URL = "https://flagcdn.com/en/codes.json"
SVG_URL_TEMPLATE = "https://flagcdn.com/{code}.svg"
PNG_URL_TEMPLATE = "https://flagcdn.com/w{width}/{code}.png"

SUPPORTED_PNG_WIDTHS = (20, 40, 80, 160, 320, 640, 1280, 2560)


@dataclass(frozen=True)
class FlagPaths:
    svg: Optional[str]
    png: Optional[str]


def _http_get_json(url: str, timeout_s: int = 30) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "InkyPi (flag downloader; +https://github.com/mcdarby/InkyPi)",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _download_to_path(url: str, dest: Path, timeout_s: int = 60, force: bool = False) -> bool:
    """
    Download URL to dest.

    Returns True if a download occurred, False if skipped due to existing file.
    """
    if dest.exists() and not force:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "InkyPi (flag downloader; +https://github.com/mcdarby/InkyPi)",
                "Accept": "*/*",
            },
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            with open(tmp_path, "wb") as f:
                f.write(response.read())
        os.replace(tmp_path, dest)
        return True
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _snap_png_width(requested_width: int) -> int:
    if requested_width in SUPPORTED_PNG_WIDTHS:
        return requested_width
    for width in SUPPORTED_PNG_WIDTHS:
        if width >= requested_width:
            return width
    return SUPPORTED_PNG_WIDTHS[-1]


def _iter_codes(
    codes: Dict[str, str],
    include_us_states: bool,
    exclude_prefixes: Iterable[str],
) -> Iterable[Tuple[str, str]]:
    for code, name in sorted(codes.items()):
        if not include_us_states and code.startswith("us-"):
            continue
        if any(code.startswith(prefix) for prefix in exclude_prefixes):
            continue
        yield code, name


def _relative_posix_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def build_manifest(
    out_dir: Path,
    flags: Dict[str, Dict[str, Any]],
    png_width: Optional[int],
    excluded_prefixes: Iterable[str],
) -> Dict[str, Any]:
    return {
        "_meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "flagcdn.com (Flagpedia)",
            "codes_url": CODES_URL,
            "formats": sorted(
                list({k for v in flags.values() for k in v.keys() if k in {"svg", "png"}})
            ),
            "png_width": png_width,
            "excluded_prefixes": list(excluded_prefixes),
        },
        "flags": flags,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download flags from flagcdn.com and generate flags.json",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["svg", "png"],
        default=["svg"],
        help="Asset formats to download (default: svg)",
    )
    parser.add_argument(
        "--png-width",
        type=int,
        default=2560,
        help="PNG width (only used if --formats includes png; default: 2560)",
    )
    parser.add_argument(
        "--no-snap-png-width",
        action="store_true",
        help="Do not snap PNG width to supported flagcdn sizes (may cause many HTTP 404s)",
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        default=[],
        help="Only download these codes (e.g. zw us gb-eng). If set, ignores prefix filters.",
    )
    parser.add_argument(
        "--include-us-states",
        action="store_true",
        help="Include US state flags (codes starting with 'us-')",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Exclude codes with this prefix (repeatable), e.g. --exclude-prefix us-",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N codes (0 = no limit)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep N seconds between downloads (default: 0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    out_dir = Path(args.out_dir).resolve()
    formats = set(args.formats)
    png_width = int(args.png_width) if "png" in formats else None
    if png_width and not args.no_snap_png_width:
        snapped = _snap_png_width(png_width)
        if snapped != png_width:
            LOGGER.warning(
                "PNG width %d is not available on flagcdn; using %d instead (supported: %s).",
                png_width,
                snapped,
                ", ".join(str(w) for w in SUPPORTED_PNG_WIDTHS),
            )
            png_width = snapped

    exclude_prefixes = list(args.exclude_prefix or [])
    if not args.include_us_states and "us-" not in exclude_prefixes:
        exclude_prefixes.append("us-")

    LOGGER.info("Fetching codes from %s", CODES_URL)
    try:
        codes = _http_get_json(CODES_URL)
    except urllib.error.URLError as e:
        LOGGER.error("Failed to fetch codes.json: %s", e)
        return 2

    if args.codes:
        requested = [c.strip().lower() for c in args.codes if c.strip()]
        missing = [c for c in requested if c not in codes]
        if missing:
            LOGGER.error("Unknown codes: %s", ", ".join(missing))
            return 2
        selected = [(code, codes[code]) for code in requested]
    else:
        selected = list(
            _iter_codes(
                codes=codes,
                include_us_states=bool(args.include_us_states),
                exclude_prefixes=exclude_prefixes,
            )
        )
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    svg_dir = out_dir / "svg"
    png_dir = out_dir / "png" / f"w{png_width}" if png_width else None

    downloaded_count = 0
    flags: Dict[str, Dict[str, Any]] = {}

    LOGGER.info("Processing %d flags into %s", len(selected), out_dir)
    for idx, (code, name) in enumerate(selected, start=1):
        paths = FlagPaths(svg=None, png=None)
        if "svg" in formats:
            svg_path = svg_dir / f"{code}.svg"
            svg_url = SVG_URL_TEMPLATE.format(code=code)
            try:
                did = _download_to_path(svg_url, svg_path, force=bool(args.force))
                downloaded_count += 1 if did else 0
                paths = FlagPaths(svg=_relative_posix_path(svg_path, out_dir), png=paths.png)
            except urllib.error.HTTPError as e:
                LOGGER.warning(
                    "(%d/%d) %s svg: HTTP %s (%s)",
                    idx,
                    len(selected),
                    code,
                    e.code,
                    svg_url,
                )
            except urllib.error.URLError as e:
                LOGGER.warning("(%d/%d) %s svg: download error: %s", idx, len(selected), code, e)

        if "png" in formats and png_width and png_dir:
            png_path = png_dir / f"{code}.png"
            png_url = PNG_URL_TEMPLATE.format(width=png_width, code=code)
            try:
                did = _download_to_path(png_url, png_path, force=bool(args.force))
                downloaded_count += 1 if did else 0
                paths = FlagPaths(svg=paths.svg, png=_relative_posix_path(png_path, out_dir))
            except urllib.error.HTTPError as e:
                LOGGER.warning(
                    "(%d/%d) %s png(w%d): HTTP %s (%s)",
                    idx,
                    len(selected),
                    code,
                    png_width,
                    e.code,
                    png_url,
                )
            except urllib.error.URLError as e:
                LOGGER.warning(
                    "(%d/%d) %s png(w%d): download error: %s",
                    idx,
                    len(selected),
                    code,
                    png_width,
                    e,
                )

        entry: Dict[str, Any] = {"name": name}
        if paths.svg:
            entry["svg"] = paths.svg
        if paths.png:
            entry["png"] = paths.png
        flags[code] = entry

        if args.sleep and args.sleep > 0:
            time.sleep(float(args.sleep))

    manifest = build_manifest(
        out_dir=out_dir,
        flags=flags,
        png_width=png_width,
        excluded_prefixes=exclude_prefixes,
    )
    manifest_path = out_dir / "flags.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    LOGGER.info("Wrote %s", manifest_path)
    LOGGER.info("Downloaded %d files", downloaded_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
