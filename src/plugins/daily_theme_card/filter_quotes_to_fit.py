#!/usr/bin/env python3
"""
Filter Daily Theme Card quote JSON so every quote fits in the left panel.

Why:
  Some quotes are too long to render legibly in the available space (e.g. 560×480
  left panel on an 800×480 device). The renderer will shrink the font down to a
  hard floor, but if the quote still doesn't fit you end up with clipped text or
  overlap with the optional top banner.

What this does:
  - Loads one or more leaf quote JSON files (format described in README.md).
  - Simulates the same wrap + font-fit logic used by `daily_theme_card.py`.
  - Removes items that do not fit within the panel bounds at/above a minimum body
    font size, and optionally reserves a top region for the banner.

Outputs:
  - Writes a filtered JSON file (in-place with --write, or to --out-dir).
  - Writes a small report JSON (optional).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.app_utils import get_font


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitResult:
    fits: bool
    body_font_size: int
    line_count: int
    total_h: int
    y0: int
    y1: int
    reason: Optional[str] = None


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font_obj) -> Tuple[int, int, int, int]:
    try:
        return draw.textbbox((0, 0), text, font=font_obj)
    except Exception:
        w2, h2 = font_obj.getsize(text)  # type: ignore[attr-defined]
        return (0, 0, w2, h2)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font_obj) -> int:
    b = _text_bbox(draw, text, font_obj)
    return int(b[2] - b[0])


def _line_height(draw: ImageDraw.ImageDraw, font_obj) -> int:
    try:
        ascent, descent = font_obj.getmetrics()
        return int(ascent + descent)
    except Exception:
        b = _text_bbox(draw, "Ag", font_obj)
        return int(b[3] - b[1])


def _wrap(draw: ImageDraw.ImageDraw, text: str, font_obj, width_limit: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        cand = f"{current} {word}"
        if _text_width(draw, cand, font_obj) <= width_limit:
            current = cand
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _panel_layout(panel_w: int, panel_h: int, draw_card_box: bool) -> Tuple[int, int, int, int]:
    """
    Returns (pad, content_x, content_max_w, card_pad) matching daily_theme_card.py.
    """
    pad = max(18, int(panel_w * 0.07))
    max_w = panel_w - pad * 2
    card_pad = max(12, int(pad * 0.55))
    content_x = pad + (card_pad if draw_card_box else 0)
    content_max_w = panel_w - (content_x * 2)
    if content_max_w < max(80, int(panel_w * 0.25)):
        content_x = pad
        content_max_w = max_w
    return pad, content_x, content_max_w, card_pad


def _fit_quote_item(
    *,
    panel_w: int,
    panel_h: int,
    draw_card_box: bool,
    text: str,
    attribution: str,
    min_body_font_size: int,
    max_body_height_ratio: float,
    reserve_top_px: int,
    reserve_bottom_px: int,
) -> FitResult:
    img = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad, _content_x, content_max_w, _card_pad = _panel_layout(panel_w, panel_h, draw_card_box)

    body_size = max(16, int(panel_w * 0.075))
    small_size = max(12, int(panel_w * 0.05))

    body_font = get_font("Jost", body_size, "normal")
    small_font = get_font("Jost", small_size, "normal")
    if body_font is None or small_font is None:
        from PIL import ImageFont

        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    max_body_h = int(panel_h * max_body_height_ratio)
    line_gap = int(max(2, pad * 0.08))

    # Mirror the renderer's sizing loop.
    for _ in range(8):
        body_lines = _wrap(draw, text, body_font, content_max_w)
        body_h = len(body_lines) * _line_height(draw, body_font)
        footer_h = (_line_height(draw, small_font) * (2 if attribution else 1)) + int(pad * 0.7)
        if body_h + footer_h <= max_body_h or body_size <= min_body_font_size:
            break
        body_size -= 2
        body_font = get_font("Jost", body_size, "normal") or body_font

    # Measure as rendered for quotes.
    body_lines = _wrap(draw, text, body_font, content_max_w)
    quote_lines = [f"“{body_lines[0]}" if body_lines else "“"]
    quote_lines += body_lines[1:]
    if quote_lines:
        quote_lines[-1] = f"{quote_lines[-1]}”"

    body_h = len(quote_lines) * (_line_height(draw, body_font) + line_gap)
    attr_h = 0
    if attribution:
        attr_h = int(pad * 0.4) + _line_height(draw, small_font)
    total_h = int(body_h + attr_h)

    y = max(pad, int((panel_h - total_h) / 2))
    y0 = y
    y1 = y + total_h

    top_limit = max(pad, int(reserve_top_px))
    bottom_limit = int(panel_h - pad - reserve_bottom_px)

    if y0 < top_limit:
        return FitResult(
            fits=False,
            body_font_size=body_size,
            line_count=len(quote_lines),
            total_h=total_h,
            y0=y0,
            y1=y1,
            reason="overlaps_reserved_top",
        )

    if y1 > bottom_limit:
        return FitResult(
            fits=False,
            body_font_size=body_size,
            line_count=len(quote_lines),
            total_h=total_h,
            y0=y0,
            y1=y1,
            reason="overflows_bottom",
        )

    # If we had to go below the minimum to satisfy the fit loop, reject.
    if body_size < min_body_font_size:
        return FitResult(
            fits=False,
            body_font_size=body_size,
            line_count=len(quote_lines),
            total_h=total_h,
            y0=y0,
            y1=y1,
            reason="requires_too_small_font",
        )

    return FitResult(
        fits=True,
        body_font_size=body_size,
        line_count=len(quote_lines),
        total_h=total_h,
        y0=y0,
        y1=y1,
        reason=None,
    )


def _iter_input_files(paths_or_globs: List[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths_or_globs:
        p = Path(raw)
        if any(ch in raw for ch in ["*", "?", "["]):
            out.extend(sorted(Path().glob(raw)))
        elif p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        else:
            out.append(p)
    # De-dupe while preserving order.
    seen: set[Path] = set()
    uniq: List[Path] = []
    for p in out:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter quote JSON items that won't fit the left panel")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSON files, directories, or glob patterns",
    )
    parser.add_argument("--panel-width", type=int, default=560, help="Left panel width in px (default: 560)")
    parser.add_argument("--panel-height", type=int, default=480, help="Left panel height in px (default: 480)")
    parser.add_argument(
        "--draw-card-box",
        action="store_true",
        help="Match background_mode != plain (slightly smaller content width)",
    )
    parser.add_argument(
        "--min-body-font-size",
        type=int,
        default=18,
        help="Reject quotes that require smaller than this size (default: 18)",
    )
    parser.add_argument(
        "--max-body-height-ratio",
        type=float,
        default=0.70,
        help="Match renderer's 'max body height' ratio (default: 0.70)",
    )
    parser.add_argument(
        "--reserve-top",
        type=int,
        default=0,
        help="Reserve N px at the top (e.g. for the banner) (default: 0)",
    )
    parser.add_argument(
        "--reserve-bottom",
        type=int,
        default=0,
        help="Reserve N px at the bottom (default: 0)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Write filtered copies to this directory (default: overwrite only with --write)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes in place (ignored if --out-dir is set)",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Write a report JSON to this path",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    inputs = _iter_input_files(args.inputs)
    if not inputs:
        LOGGER.error("No input files found.")
        return 2

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "params": {
            "panel_width": args.panel_width,
            "panel_height": args.panel_height,
            "draw_card_box": bool(args.draw_card_box),
            "min_body_font_size": int(args.min_body_font_size),
            "max_body_height_ratio": float(args.max_body_height_ratio),
            "reserve_top": int(args.reserve_top),
            "reserve_bottom": int(args.reserve_bottom),
        },
        "files": [],
    }

    for path in inputs:
        if not path.exists() or path.suffix.lower() != ".json":
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("items")
        if not isinstance(items, list):
            LOGGER.info("Skipping (no items list): %s", path)
            continue

        card_type = str(data.get("type") or "quote").strip().lower()
        if card_type != "quote":
            LOGGER.info("Skipping (type=%s): %s", card_type, path)
            continue

        kept: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            attribution = str(item.get("attribution") or "").strip()
            if not text:
                continue

            fit = _fit_quote_item(
                panel_w=int(args.panel_width),
                panel_h=int(args.panel_height),
                draw_card_box=bool(args.draw_card_box),
                text=text,
                attribution=attribution,
                min_body_font_size=int(args.min_body_font_size),
                max_body_height_ratio=float(args.max_body_height_ratio),
                reserve_top_px=int(args.reserve_top),
                reserve_bottom_px=int(args.reserve_bottom),
            )

            item_out = dict(item)
            item_out["layout"] = {
                "fits": fit.fits,
                "body_font_size": fit.body_font_size,
                "line_count": fit.line_count,
                "reason": fit.reason,
            }
            if fit.fits:
                kept.append(item_out)
            else:
                removed.append(item_out)

        data["items"] = kept

        file_report = {
            "path": str(path),
            "items_in": len(items),
            "items_out": len(kept),
            "removed": len(removed),
            "removed_reasons": {},
        }
        reasons: Dict[str, int] = {}
        for it in removed:
            reason = (it.get("layout") or {}).get("reason") or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
        file_report["removed_reasons"] = reasons
        report["files"].append(file_report)

        LOGGER.info(
            "%s: kept %d/%d (removed %d)",
            path.name,
            len(kept),
            len(items),
            len(removed),
        )

        if out_dir:
            out_path = out_dir / path.name
            out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif args.write:
            path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.report:
        report_path = Path(args.report).resolve()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOGGER.info("Wrote report: %s", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
