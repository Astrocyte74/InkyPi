from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, time, timedelta

import pytz
from PIL import Image, ImageDraw

from plugins.base_plugin.base_plugin import BasePlugin
from utils.app_utils import get_font, resolve_path
from utils.family_banner import draw_family_banner
from utils.family_events import banner_from_env
from utils.openweather import fetch_weather_snapshot
from utils.weather_sidebar import render_weather_sidebar_panel

logger = logging.getLogger(__name__)


CFM_DATA_DIR = resolve_path(os.path.join("plugins", "cfm_2026", "data"))
SIDEBAR_WIDTH_RATIO = 0.30
QUOTE_MIN_BODY_FONT_SIZE = 18
BANNER_RESERVE_TOP_PX = 80
DAILY_REFRESH_TIME_DEFAULT = "04:00"


class Cfm2026(BasePlugin):
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

        daily_refresh_time = self._parse_hhmm(settings.get("dailyRefreshTime") or DAILY_REFRESH_TIME_DEFAULT)

        background_mode = (settings.get("backgroundMode") or "plain").strip().lower()
        if background_mode not in {"plain", "illustration", "illustration_blur"}:
            background_mode = "plain"
        illustration_cache_id = (settings.get("illustrationCacheId") or "").strip()

        tz_str = device_config.get_config("timezone", default="UTC")
        tz = pytz.timezone(tz_str)
        now = datetime.now(tz)

        dimensions = device_config.get_resolution()
        if device_config.get_config("orientation") == "vertical":
            dimensions = dimensions[::-1]

        width, height = dimensions
        sidebar_width = int(width * SIDEBAR_WIDTH_RATIO)
        sidebar_width = max(120, min(width - 100, sidebar_width))
        image_width = width - sidebar_width
        sidebar_size_px = (sidebar_width, height)

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

        day_key = self._day_key(now, daily_refresh_time)

        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        base = None
        if background_mode in {"illustration", "illustration_blur"}:
            base = self._load_illustration_background(
                device_config,
                size=(image_width, height),
                cache_id=illustration_cache_id,
                blur=(background_mode == "illustration_blur"),
            )

        banner = None
        try:
            banner = banner_from_env(device_config, now=now)
        except Exception:
            logger.exception("Failed to compute family banner; continuing.")

        # Load the current week's data
        week_data = self._load_week_data(now)
        week_number = week_data.get("id", "")
        # No title header - quote speaks for itself (like Daily Theme Card)
        title = ""
        subtitle = ""
        items = week_data.get("items", [])

        # Pick a quote for the day
        quote_item = self._pick_quote(items, week_number, day_key)
        text = quote_item.get("text", "")
        attribution = quote_item.get("attribution", "")

        left = self._render_left_panel(
            title=title,
            subtitle=subtitle,
            text=text,
            attribution=attribution,
            size=(image_width, height),
            base=base,
            reserve_top_px=(BANNER_RESERVE_TOP_PX if banner else 0),
        )

        try:
            if banner:
                draw_family_banner(
                    left,
                    headline=banner.get("headline") or "",
                    detail=banner.get("detail") or "",
                    font_fn=self._font,
                    region=(0, 0, image_width, height),
                )
        except Exception:
            logger.exception("Failed to render family banner; continuing.")

        canvas.paste(left, (0, 0))

        panel = render_weather_sidebar_panel(
            weather,
            tz,
            forecast_days,
            sidebar_size_px,
            location_label,
            font=self._font,
            icon_renderer=self._simple_weather_icon,
        )
        canvas.paste(panel, (image_width, 0))
        return canvas

    @staticmethod
    def _parse_hhmm(value):
        value = (value or "").strip()
        import re

        match = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
        if not match:
            return time(4, 0)
        hour = int(match.group(1))
        minute = int(match.group(2))
        hour = max(0, min(23, hour))
        minute = max(0, min(59, minute))
        return time(hour, minute)

    @staticmethod
    def _day_key(now, refresh_time):
        if now.timetz().replace(tzinfo=None) < refresh_time:
            day = (now - timedelta(days=1)).date()
        else:
            day = now.date()
        return day.isoformat()

    @classmethod
    def _load_week_data(cls, now):
        """Load the JSON file for the current week based on date."""
        # CFM weeks typically start on Monday
        # Find the Monday of the current week
        year = now.year
        jan_1 = datetime(year, 1, 1)
        # Get the first Monday of the year (week 1 starts on a Monday near Jan 1)
        days_to_monday = (0 - jan_1.weekday()) % 7
        first_monday = jan_1 + timedelta(days=days_to_monday)

        # Calculate week number (1-53)
        days_since_first_monday = (now.date() - first_monday.date()).days
        week_number = max(1, (days_since_first_monday // 7) + 1)

        # Handle edge case for week 53 (partial week at year end)
        if week_number > 53:
            week_number = 1
            year += 1

        week_id = f"{year}-{week_number:02d}"
        json_path = os.path.join(CFM_DATA_DIR, f"{week_id}.json")

        # Try to load the JSON file
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load week JSON %s: %s", json_path, e)

        # Fallback: try to find any valid week JSON
        try:
            if os.path.isdir(CFM_DATA_DIR):
                for filename in sorted(os.listdir(CFM_DATA_DIR)):
                    if filename.endswith(".json"):
                        fallback_path = os.path.join(CFM_DATA_DIR, filename)
                        try:
                            with open(fallback_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                logger.info("Using fallback JSON: %s", filename)
                                return data
                        except Exception:
                            continue
        except Exception:
            pass

        # Ultimate fallback
        return {
            "id": week_id,
            "label": f"Week {week_number}",
            "title": "Come, Follow Me",
            "subtitle": "",
            "items": [{"text": "Unable to load lesson data.", "attribution": ""}]
        }

    @classmethod
    def _per_refresh_bucket(cls, *, week_id: str, day_key: str) -> int:
        """Get a counter that increments on each refresh for this week/day."""
        seed = f"{week_id}|{day_key}"
        key = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        cls._PER_REFRESH_COUNTER[key] = cls._PER_REFRESH_COUNTER.get(key, -1) + 1
        return cls._PER_REFRESH_COUNTER[key]

    @classmethod
    def _pick_quote(cls, items, week_id, day_key):
        """Pick a quote that cycles through on each refresh."""
        if not items or not isinstance(items, list):
            return {"text": "No items configured.", "attribution": ""}

        # Use per-refresh bucket to cycle through all quotes
        bucket = cls._per_refresh_bucket(week_id=week_id, day_key=day_key)
        idx = bucket % len(items)
        return items[idx]

    @classmethod
    def _daily_cat_cache_dir(cls, device_config):
        return os.path.join(device_config.BASE_DIR, "..", "mock_display_output", "daily_cat_weather")

    @staticmethod
    def _cover_crop(img, size):
        from PIL import ImageOps
        return ImageOps.fit(img, size, method=Image.LANCZOS, centering=(0.5, 0.5))

    @classmethod
    def _find_latest_cat_bg(cls, device_config, *, cache_id=""):
        cache_dir = cls._daily_cat_cache_dir(device_config)
        cache_id = (cache_id or "").strip()
        if cache_id:
            candidate = os.path.join(cache_dir, f"latest_bg_{cache_id}.png")
            if os.path.exists(candidate):
                return candidate

        if not os.path.isdir(cache_dir):
            return ""

        latest_path = ""
        latest_mtime = -1.0
        try:
            for name in os.listdir(cache_dir):
                if not name.startswith("latest_bg_") or not name.endswith(".png"):
                    continue
                path = os.path.join(cache_dir, name)
                try:
                    mtime = os.path.getmtime(path)
                except Exception:
                    continue
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = path
        except Exception:
            return ""

        return latest_path

    @classmethod
    def _load_illustration_background(cls, device_config, *, size, cache_id="", blur=False):
        from PIL import ImageEnhance, ImageFilter

        path = cls._find_latest_cat_bg(device_config, cache_id=cache_id)
        if not path:
            return None
        try:
            with Image.open(path) as img:
                bg = img.convert("RGB")
        except Exception:
            logger.exception("Failed to load illustration background: %s", path)
            return None

        if bg.size != size:
            bg = cls._cover_crop(bg, size)

        if blur:
            radius = max(2, min(14, int(size[0] * 0.018)))
            bg = bg.filter(ImageFilter.GaussianBlur(radius=radius))
            bg = ImageEnhance.Brightness(bg).enhance(0.72)
            bg = ImageEnhance.Color(bg).enhance(0.60)
        else:
            bg = ImageEnhance.Brightness(bg).enhance(0.86)
            bg = ImageEnhance.Color(bg).enhance(0.85)

        return bg

    def _render_left_panel(
        self,
        *,
        title,
        subtitle,
        text,
        attribution,
        size,
        base=None,
        reserve_top_px: int = 0,
    ):
        """Render the quote panel, matching Daily Theme Card styling exactly."""
        w, h = size
        if base is not None:
            img = base.copy().convert("RGB")
        else:
            img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        pad = max(18, int(w * 0.07))
        max_w = w - pad * 2
        card_pad = max(12, int(pad * 0.55))
        content_x = pad + card_pad
        content_max_w = w - (content_x * 2)

        small_font = self._font("Jost", max(12, int(w * 0.05)))

        def text_bbox(text, font_obj):
            try:
                return draw.textbbox((0, 0), text, font=font_obj)
            except Exception:
                w2, h2 = font_obj.getsize(text)  # type: ignore[attr-defined]
                return (0, 0, w2, h2)

        def text_width(text, font_obj):
            b = text_bbox(text, font_obj)
            return b[2] - b[0]

        def line_height(font_obj):
            try:
                ascent, descent = font_obj.getmetrics()
                return int(ascent + descent)
            except Exception:
                return text_bbox("Ag", font_obj)[3]

        def wrap(text, font_obj, width_limit):
            words = (text or "").split()
            if not words:
                return []
            lines = []
            current = words[0]
            for word in words[1:]:
                cand = f"{current} {word}"
                if text_width(cand, font_obj) <= width_limit:
                    current = cand
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
            return lines

        # Draw "Come, Follow Me" title pill at the top (will be covered by banner if present)
        title_pill_text = "Come, Follow Me"
        title_pill_font = self._font("Jost", max(12, int(w * 0.045)), bold=True)
        title_pill_pad = max(8, int(w * 0.025))
        title_pill_height = line_height(title_pill_font) + title_pill_pad * 2

        title_pill_tb = text_bbox(title_pill_text, title_pill_font)
        title_pill_w = title_pill_tb[2] - title_pill_tb[0] + title_pill_pad * 2
        title_pill_x = (w - title_pill_w) // 2
        title_pill_y = max(6, int(w * 0.015))

        # Draw pill background
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        radius = max(6, int(title_pill_height * 0.4))
        try:
            od.rounded_rectangle(
                (title_pill_x, title_pill_y, title_pill_x + title_pill_w, title_pill_y + title_pill_height),
                radius=radius,
                fill=(255, 255, 255, 220),
                outline=(0, 0, 0, 80),
                width=1,
            )
        except Exception:
            od.rectangle(
                (title_pill_x, title_pill_y, title_pill_x + title_pill_w, title_pill_y + title_pill_height),
                fill=(255, 255, 255, 220),
            )
        img_rgba = img.convert("RGBA")
        img_rgba.alpha_composite(overlay)
        img = img_rgba.convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw title text centered in pill
        title_text_x = title_pill_x + title_pill_pad
        title_text_y = title_pill_y + title_pill_pad
        draw.text((title_text_x, title_text_y), title_pill_text, fill=(0, 0, 0), font=title_pill_font)

        # Reserve space for title pill (banner will overlay this if present)
        title_pill_reserve = title_pill_y + title_pill_height + max(6, int(w * 0.015))

        def draw_card_box(y0, total_h):
            """Draw a white rounded rectangle card behind the quote."""
            x0 = pad
            x1 = w - pad
            y_top = max(pad, int(y0 - card_pad))
            y_bot = min(h - pad, int(y0 + total_h + card_pad))
            overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            radius = max(10, int(min(x1 - x0, y_bot - y_top) * 0.06))
            try:
                od.rounded_rectangle((x0, y_top, x1, y_bot), radius=radius, fill=(255, 255, 255, 235), outline=(0, 0, 0, 110), width=2)
            except Exception:
                od.rectangle((x0, y_top, x1, y_bot), fill=(255, 255, 255, 235))
            img_rgba = img.convert("RGBA")
            img_rgba.alpha_composite(overlay)
            return img_rgba.convert("RGB")

        # Smart initial font size based on character count
        # Short quotes get larger fonts, long quotes start smaller
        char_count = len(text)
        base_size = max(16, int(w * 0.075))
        if char_count > 200:
            # Reduce font size for longer quotes (every 120 chars over 200 = 1px smaller)
            reduction = (char_count - 200) // 120
            body_size = max(QUOTE_MIN_BODY_FONT_SIZE, base_size - reduction)
        else:
            body_size = base_size
        body_font = self._font("Jost", body_size)

        # Fit body font to available space (title pill is at top, not in card)
        max_body_h = int(h * 0.70)
        for _ in range(12):  # More iterations for better fit
            body_lines = wrap(text, body_font, content_max_w)
            body_h = len(body_lines) * line_height(body_font)
            footer_h = (line_height(small_font) * (2 if attribution else 1)) + int(pad * 0.7)
            if body_h + footer_h <= max_body_h or body_size <= QUOTE_MIN_BODY_FONT_SIZE:
                break
            body_size -= 2
            body_font = self._font("Jost", body_size)

        # Wrap quote with straight quotes (like Daily Theme Card)
        body_lines = wrap(text, body_font, content_max_w)
        quote_lines = [f"'{body_lines[0]}" if body_lines else "'"]
        quote_lines += body_lines[1:]
        if quote_lines:
            quote_lines[-1] = f"{quote_lines[-1]}"

        # Calculate total height (no title in card, it's at top in pill)
        line_gap = int(max(2, pad * 0.08))
        body_h = len(quote_lines) * (line_height(body_font) + line_gap)
        attr_h = 0
        if attribution:
            attr_h = int(pad * 0.4) + line_height(small_font)
        total_h = body_h + attr_h

        # Center the card vertically in the panel (accounting for title pill and banner)
        top_limit = max(pad, title_pill_reserve, int(reserve_top_px or 0))
        available_h = h - top_limit - pad
        y = top_limit + max(pad, int((available_h - total_h) / 2))

        # Draw the white card box
        boxed = draw_card_box(y, total_h)
        if boxed is not None:
            img = boxed
            draw = ImageDraw.Draw(img)

        # Draw quote lines, centered horizontally within the card
        content_left = content_x
        content_right = w - content_x
        inner_w = max(1, content_right - content_left)

        for line in quote_lines:
            x = content_left + max(0, int((inner_w - text_width(line, body_font)) / 2))
            draw.text((x, y), line, fill=(0, 0, 0), font=body_font)
            y += line_height(body_font) + line_gap

        # Draw attribution, centered
        if attribution:
            y += int(pad * 0.4)
            attr_line = f"— {attribution}"
            x = content_left + max(0, int((inner_w - text_width(attr_line, small_font)) / 2))
            draw.text((x, y), attr_line, fill=(0, 0, 0), font=small_font)

        return img

    @staticmethod
    def _font(family, size, bold=False):
        font = get_font(family, font_size=size, font_weight="bold" if bold else "normal")
        if font is None:
            from PIL import ImageFont
            return ImageFont.load_default()
        return font

    @staticmethod
    def _simple_weather_icon(icon_code, size):
        from plugins.daily_cat_weather.daily_cat_weather import DailyCatWeather
        return DailyCatWeather._simple_weather_icon(icon_code, size)  # pylint: disable=protected-access
