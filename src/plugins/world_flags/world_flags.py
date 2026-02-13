from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta

import pytz
from PIL import Image, ImageDraw, ImageFont, ImageOps

from plugins.base_plugin.base_plugin import BasePlugin
from utils.family_banner import draw_family_banner
from utils.family_events import banner_from_env
from utils.openweather import fetch_weather_snapshot
from utils.weather_sidebar import render_weather_sidebar_panel

logger = logging.getLogger(__name__)


FLAGS_PATH_DEFAULT = os.path.join("plugins", "daily_theme_card", "flags", "flags.json")
SIDEBAR_WIDTH_RATIO = 0.30


@dataclass(frozen=True)
class FlagEntry:
    code: str
    name: str
    png_path: str
    title: str
    lines: list[str]


class WorldFlags(BasePlugin):
    """Render a world flag + small info box in the left panel with the standard weather sidebar."""

    _FLAGS_CACHE: tuple[str, float, list[FlagEntry]] | None = None
    _PER_REFRESH_COUNTER: dict[str, int] = {}

    def generate_settings_template(self):
        template_params = super().generate_settings_template()
        template_params["api_key"] = {
            "required": True,
            "service": "OpenWeatherMap",
            "expected_key": "OPEN_WEATHER_MAP_SECRET",
        }
        return template_params

    def generate_image(self, settings, device_config):
        owm_key = device_config.load_env_key("OPEN_WEATHER_MAP_SECRET")
        if not owm_key:
            raise RuntimeError("OPEN_WEATHER_MAP_SECRET is not configured.")

        lat = (settings.get("latitude") or "").strip()
        lon = (settings.get("longitude") or "").strip()
        if not lat or not lon:
            raise RuntimeError("Latitude and Longitude are required.")

        location_label = (settings.get("locationLabel") or "").strip()

        units = (settings.get("units") or "metric").strip().lower()
        if units not in {"metric", "imperial", "standard"}:
            raise RuntimeError("Units must be one of metric, imperial, or standard.")

        forecast_days = int(settings.get("forecastDays") or 3)
        forecast_days = max(1, min(5, forecast_days))

        weather_cache_minutes = int(settings.get("weatherCacheMinutes") or 30)
        weather_cache_minutes = max(0, min(1440, weather_cache_minutes))

        tz_str = device_config.get_config("timezone", default="UTC")
        tz = pytz.timezone(tz_str)
        now = datetime.now(tz)

        daily_refresh_time = self._parse_hhmm(settings.get("dailyRefreshTime") or "04:00")
        day_key = self._day_key(now, daily_refresh_time)

        rotation_mode = self._rotation_mode(settings)
        rotation_period = self._rotation_period_minutes(settings)
        if rotation_mode in {"per_refresh", "random_per_refresh"}:
            bucket = self._per_refresh_bucket(day_key=day_key, settings=settings)
        else:
            bucket = self._rotation_bucket(now, day_key, daily_refresh_time, tz, rotation_mode, rotation_period)

        flags_path = (settings.get("flagsPath") or "").strip() or FLAGS_PATH_DEFAULT
        entries = self._load_flags(flags_path, device_config)
        if not entries:
            raise RuntimeError("No flags available (flags.json missing or empty).")

        include_codes = self._parse_codes(settings.get("includeCodes"))
        exclude_codes = self._parse_codes(settings.get("excludeCodes"))
        if include_codes:
            entries = [e for e in entries if e.code in include_codes]
        if exclude_codes:
            entries = [e for e in entries if e.code not in exclude_codes]
        if not entries:
            raise RuntimeError("No flags match include/exclude filters.")

        chosen = self._pick_entry(entries, day_key, bucket, rotation_mode)

        dimensions = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dimensions = dimensions[::-1]
        width, height = dimensions

        sidebar_width = int(width * SIDEBAR_WIDTH_RATIO)
        sidebar_width = max(120, min(width - 100, sidebar_width))
        left_w = width - sidebar_width

        weather = None
        try:
            weather = fetch_weather_snapshot(
                api_key=owm_key,
                units=units,
                lat=lat,
                lon=lon,
                now=now,
                cache_ttl_sec=weather_cache_minutes * 60,
            )
        except Exception:
            logger.exception("Failed to fetch weather; continuing without sidebar data.")

        left = self._render_left_panel(chosen, size=(left_w, height))
        try:
            banner = banner_from_env(device_config, now=now)
            if banner:
                draw_family_banner(
                    left,
                    headline=banner.get("headline") or "",
                    detail=banner.get("detail") or "",
                    font_fn=self._font,
                    region=(0, 0, left_w, height),
                )
        except Exception:
            logger.exception("Failed to render family banner; continuing.")

        panel = render_weather_sidebar_panel(
            weather,
            tz,
            forecast_days,
            (sidebar_width, height),
            location_label,
            font=self._font,
            icon_renderer=self._simple_weather_icon,
        )

        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(left, (0, 0))
        canvas.paste(panel, (left_w, 0))
        return canvas

    @staticmethod
    def _parse_hhmm(value):
        value = (value or "").strip()
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
        if not match:
            return time(4, 0)
        hour = max(0, min(23, int(match.group(1))))
        minute = max(0, min(59, int(match.group(2))))
        return time(hour, minute)

    @staticmethod
    def _day_key(now: datetime, refresh_time: time) -> str:
        if now.timetz().replace(tzinfo=None) < refresh_time:
            day = (now - timedelta(days=1)).date()
        else:
            day = now.date()
        return day.isoformat()

    @staticmethod
    def _rotation_mode(settings):
        mode = str((settings or {}).get("rotationMode") or "daily").strip().lower()
        if mode in {"seq", "sequence"}:
            mode = "sequential"
        if mode in {"randomperrefresh", "random_per_refresh", "random-refresh", "randomrefresh"}:
            mode = "random_per_refresh"
        if mode in {"perrefresh", "per_refresh", "refresh"}:
            mode = "per_refresh"
        if mode not in {"daily", "sequential", "random", "per_refresh", "random_per_refresh"}:
            mode = "daily"
        return mode

    @staticmethod
    def _rotation_period_minutes(settings):
        raw = (settings or {}).get("rotationPeriodMinutes")
        try:
            minutes = int(str(raw).strip())
        except Exception:
            minutes = 60
        return max(1, min(1440, minutes))

    @classmethod
    def _per_refresh_bucket(cls, *, day_key: str, settings: dict) -> int:
        seed = json.dumps(
            {
                "day_key": day_key,
                "include": settings.get("includeCodes"),
                "exclude": settings.get("excludeCodes"),
                "flagsPath": settings.get("flagsPath"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        key = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        cls._PER_REFRESH_COUNTER[key] = cls._PER_REFRESH_COUNTER.get(key, -1) + 1
        return cls._PER_REFRESH_COUNTER[key]

    @staticmethod
    def _rotation_bucket(now: datetime, day_key: str, refresh_time: time, tz, mode: str, period_minutes: int) -> int:
        if mode == "daily":
            return 0
        if mode in {"per_refresh", "random_per_refresh"}:
            return 0
        try:
            day = datetime.strptime(day_key, "%Y-%m-%d").date()
            anchor = tz.localize(datetime.combine(day, refresh_time))
            minutes_since = max(0, int((now - anchor).total_seconds() // 60))
            return minutes_since // period_minutes
        except Exception:
            return 0

    @staticmethod
    def _parse_codes(value):
        if value is None:
            return []
        if isinstance(value, list):
            raw = ",".join(str(v) for v in value)
        else:
            raw = str(value)
        codes = []
        for token in re.split(r"[,\s]+", raw.strip()):
            token = token.strip().lower()
            if not token:
                continue
            if len(token) == 2 and token.isalpha():
                codes.append(token)
        # de-dupe preserve order
        seen = set()
        out = []
        for c in codes:
            if c in seen:
                continue
            seen.add(c)
            out.append(c)
        return out

    @classmethod
    def _load_flags(cls, flags_path: str, device_config) -> list[FlagEntry]:
        try:
            from utils.app_utils import resolve_path  # local import
        except Exception:
            resolve_path = None

        path = flags_path
        if not os.path.isabs(path) and resolve_path:
            path = resolve_path(path)

        try:
            mtime = os.path.getmtime(path)
        except Exception:
            return []

        cache = cls._FLAGS_CACHE
        if cache and cache[0] == path and cache[1] == mtime:
            return cache[2]

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle) or {}
        except Exception:
            logger.exception("Failed to load flags manifest: %s", path)
            return []

        flags = payload.get("flags") if isinstance(payload, dict) else None
        if not isinstance(flags, dict):
            return []

        base_dir = os.path.dirname(path)
        entries: list[FlagEntry] = []
        for code, entry in flags.items():
            if not isinstance(entry, dict):
                continue
            flag_obj = entry.get("flag") if isinstance(entry.get("flag"), dict) else {}
            png_rel = (flag_obj.get("png") or entry.get("png") or "").strip()
            if not png_rel:
                continue
            png_path = os.path.join(base_dir, png_rel)
            if not os.path.exists(png_path):
                continue
            name = str(entry.get("name") or code).strip()
            display = entry.get("display") if isinstance(entry.get("display"), dict) else {}
            title = str(display.get("title") or name).strip() or name
            lines = display.get("lines") if isinstance(display.get("lines"), list) else []
            lines = [str(x).strip() for x in lines if str(x).strip()]
            if not lines:
                # fallback to basic facts
                capital = str(entry.get("capital") or "").strip()
                pop = str(entry.get("population_display") or "").strip()
                area = str(entry.get("area_vs_alberta_line") or "").strip()
                if capital:
                    lines.append(f"Capital: {capital}")
                if pop:
                    lines.append(f"Population: {pop}")
                if area:
                    lines.append(area if area.lower().startswith("area:") else f"Area: {area}")
                lines = lines[:3]
            entries.append(
                FlagEntry(
                    code=str(code).strip().lower(),
                    name=name,
                    png_path=png_path,
                    title=title,
                    lines=lines[:3],
                )
            )

        entries.sort(key=lambda e: e.code)
        cls._FLAGS_CACHE = (path, mtime, entries)
        return entries

    @staticmethod
    def _pick_entry(entries: list[FlagEntry], day_key: str, bucket: int, mode: str) -> FlagEntry:
        if not entries:
            raise RuntimeError("No flags available.")
        if len(entries) == 1:
            return entries[0]

        if mode == "per_refresh":
            idx = bucket % len(entries)
            return entries[idx]

        if mode == "random_per_refresh":
            seed = f"{day_key}|{bucket}|{mode}".encode("utf-8")
            idx = int(hashlib.sha256(seed).hexdigest(), 16) % len(entries)
            return entries[idx]

        if mode == "sequential":
            idx = bucket % len(entries)
            return entries[idx]

        seed = f"{day_key}|{bucket}|{mode}".encode("utf-8")
        idx = int(hashlib.sha256(seed).hexdigest(), 16) % len(entries)
        return entries[idx]

    @staticmethod
    def _fit_contain(img: Image.Image, size: tuple[int, int]) -> Image.Image:
        return ImageOps.contain(img, size, method=Image.LANCZOS)

    def _render_left_panel(self, entry: FlagEntry, *, size: tuple[int, int]) -> Image.Image:
        w, h = size
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Unified frame layout - single border around flag + info
        frame_pad = max(12, int(w * 0.025))
        frame_border = 4  # Thicker border for e-ink visibility

        # Inner content area (accounting for frame border)
        inner_x0 = frame_pad
        inner_y0 = frame_pad
        inner_x1 = w - frame_pad
        inner_y1 = h - frame_pad
        inner_w = inner_x1 - inner_x0
        inner_h = inner_y1 - inner_y0

        # Calculate flag and info sections within the unified frame
        flag_h = int(inner_h * 0.65)  # Flag takes 65% of height
        info_h = inner_h - flag_h
        divider_y = inner_y0 + flag_h

        # Draw the unified frame (single border around everything)
        frame_rect = (inner_x0, inner_y0, inner_x1, inner_y1)
        draw.rectangle(frame_rect, outline=(0, 0, 0), width=frame_border)

        # Draw divider line between flag and info
        divider_pad = frame_border + 4
        draw.line(
            [(inner_x0 + divider_pad, divider_y), (inner_x1 - divider_pad, divider_y)],
            fill=(0, 0, 0),
            width=2
        )

        # Flag region (inside the frame, above divider)
        flag_region = (
            inner_x0 + frame_border,
            inner_y0 + frame_border,
            inner_x1 - frame_border,
            divider_y - 4
        )

        # Load and render flag image
        try:
            with Image.open(entry.png_path) as im:
                flag = im.convert("RGB")
        except Exception:
            logger.exception("Failed to load flag image: %s", entry.png_path)
            flag = None

        if flag is not None:
            fw = max(10, flag_region[2] - flag_region[0])
            fh = max(10, flag_region[3] - flag_region[1])
            fitted = self._fit_contain(flag, (fw, fh))
            fx = flag_region[0] + (fw - fitted.size[0]) // 2
            fy = flag_region[1] + (fh - fitted.size[1]) // 2
            img.paste(fitted, (fx, fy))

        # Info section (below divider, inside frame)
        info_x0 = inner_x0 + frame_border
        info_y0 = divider_y + 8
        info_x1 = inner_x1 - frame_border
        info_y1 = inner_y1 - frame_border
        info_w = info_x1 - info_x0
        info_h = info_y1 - info_y0

        # Font sizing
        title_font_size = max(18, int(min(w, h) * 0.055))
        line_font_size = max(13, int(min(w, h) * 0.042))
        title_font = self._font("Jost", title_font_size, bold=True)
        line_font = self._font("Jost", line_font_size, bold=False)

        text_pad_x = max(8, int(line_font_size * 0.6))
        text_pad_y = max(6, int(line_font_size * 0.5))
        max_text_w = info_w - text_pad_x * 2

        def text_w(txt: str, font_obj: ImageFont.ImageFont) -> int:
            try:
                return int(draw.textlength(txt, font=font_obj))
            except Exception:
                bbox = draw.textbbox((0, 0), txt, font=font_obj)
                return max(0, bbox[2] - bbox[0])

        def truncate(txt: str, font_obj: ImageFont.ImageFont) -> str:
            txt = (txt or "").strip()
            if not txt:
                return ""
            if text_w(txt, font_obj) <= max_text_w:
                return txt
            ell = "…"
            lo, hi = 0, len(txt)
            best = ell
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = txt[:mid].rstrip() + ell
                if text_w(cand, font_obj) <= max_text_w:
                    best = cand
                    lo = mid + 1
                else:
                    hi = mid - 1
            return best

        # Render info text
        y = info_y0 + text_pad_y
        title = truncate(entry.title, title_font)
        draw.text((info_x0 + text_pad_x, y), title, font=title_font, fill=(0, 0, 0))
        y += int(title_font_size * 1.20)

        for line in entry.lines[:3]:
            line = truncate(line, line_font)
            draw.text((info_x0 + text_pad_x, y), line, font=line_font, fill=(0, 0, 0))
            y += int(line_font_size * 1.25)

        return img

    @staticmethod
    def _font(family, size, bold=False):
        from utils.app_utils import get_font  # local import to avoid circular imports

        return get_font(family, font_size=size, font_weight="bold" if bold else "normal")

    @staticmethod
    def _simple_weather_icon(icon_code, size):
        # Use the same simple icon renderer as other plugins if available.
        try:
            from plugins.daily_cat_weather.daily_cat_weather import DailyCatWeather  # local import

            return DailyCatWeather._simple_weather_icon(icon_code, size)  # pylint: disable=protected-access
        except Exception:
            return Image.new("RGBA", (int(size), int(size)), (0, 0, 0, 0))
