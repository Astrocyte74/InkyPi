from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, time, timedelta

import pytz
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from plugins.base_plugin.base_plugin import BasePlugin
from utils.app_utils import get_font, resolve_path
from utils.family_banner import draw_family_banner
from utils.family_events import banner_from_env
from utils.openweather import fetch_weather_snapshot
from utils.weather_sidebar import render_weather_sidebar_panel

logger = logging.getLogger(__name__)


CARDS_PATH = resolve_path(os.path.join("plugins", "daily_theme_card", "cards.json"))
CARDS_DIR = os.path.dirname(CARDS_PATH)
SIDEBAR_WIDTH_RATIO = 0.30
DEFAULT_CARD_ID = "inspiration"


class DailyThemeCard(BasePlugin):
    def generate_settings_template(self):
        template_params = super().generate_settings_template()
        template_params["api_key"] = {
            "required": True,
            "service": "OpenWeatherMap",
            "expected_key": "OPEN_WEATHER_MAP_SECRET",
        }
        template_params["card_leaf_choices"] = self._card_leaf_choices()
        template_params["card_groups"] = self._card_groups()
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

        card_id = self._normalize_id(settings.get("cardId") or DEFAULT_CARD_ID) or DEFAULT_CARD_ID

        background_mode = (settings.get("backgroundMode") or "illustration_blur").strip().lower()
        if background_mode not in {"plain", "illustration", "illustration_blur"}:
            background_mode = "illustration_blur"
        illustration_cache_id = (settings.get("illustrationCacheId") or "").strip()

        daily_refresh_time = self._parse_hhmm(settings.get("dailyRefreshTime") or "04:00")
        tz_str = device_config.get_config("timezone", default="UTC")
        tz = pytz.timezone(tz_str)
        now = datetime.now(tz)
        day_key = self._day_key(now, daily_refresh_time)

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
            weather = fetch_weather_snapshot(api_key=owm_key, units=units, lat=lat, lon=lon, now=now)
        except Exception:
            logger.exception("Failed to fetch weather; continuing without sidebar data.")

        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        base = None
        if background_mode in {"illustration", "illustration_blur"}:
            base = self._load_illustration_background(
                device_config,
                size=(image_width, height),
                cache_id=illustration_cache_id,
                blur=(background_mode == "illustration_blur"),
            )
        left = self._render_card_left_panel(
            card_id=card_id,
            day_key=day_key,
            size=(image_width, height),
            base=base,
            draw_card_box=(background_mode != "plain"),
        )
        try:
            banner = banner_from_env(device_config, now=now)
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
    def _daily_cat_cache_dir(device_config):
        return os.path.join(device_config.BASE_DIR, "..", "mock_display_output", "daily_cat_weather")

    @staticmethod
    def _cover_crop(img, size):
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

    @staticmethod
    def _normalize_id(value):
        return re.sub(r"[^a-z0-9_]+", "", (value or "").strip().lower())

    @staticmethod
    def _parse_hhmm(value):
        value = (value or "").strip()
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
    def _read_json(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def _resolve_source(cls, source):
        source = (source or "").strip()
        if not source:
            return ""
        if os.path.isabs(source):
            candidate = os.path.abspath(source)
            allowed = os.path.abspath("/usr/local/inkypi")
            if os.path.commonpath([allowed, candidate]) != allowed:
                raise ValueError(f"Invalid absolute source path: {source}")
            return candidate
        candidate = os.path.abspath(os.path.join(CARDS_DIR, source))
        base = os.path.abspath(CARDS_DIR)
        if os.path.commonpath([base, candidate]) != base:
            raise ValueError(f"Invalid source path outside plugin directory: {source}")
        return candidate

    @classmethod
    def _clean_quote_or_word_items(cls, items):
        if not isinstance(items, list):
            return []
        cleaned = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            attribution = str(item.get("attribution") or "").strip()
            cleaned.append({"text": text, "attribution": attribution})
        return cleaned

    @classmethod
    def _clean_family_items(cls, items):
        if not isinstance(items, list):
            return []
        cleaned = []
        for item in items:
            if not isinstance(item, dict):
                continue
            date_str = str(item.get("date") or "").strip()
            if not re.fullmatch(r"\d{2}-\d{2}", date_str):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            year = item.get("year")
            year_int = None
            if isinstance(year, int):
                year_int = year
            elif isinstance(year, str) and year.strip().isdigit():
                year_int = int(year.strip())
            cleaned.append({"date": date_str, "text": text, "year": year_int})
        return cleaned

    @classmethod
    def _clean_card_payload(cls, payload, *, fallback_id=""):
        if not isinstance(payload, dict):
            return None
        card_id = cls._normalize_id(payload.get("id") or fallback_id)
        if not card_id:
            return None
        label = str(payload.get("label") or card_id).strip() or card_id
        card_type = str(payload.get("type") or "quote").strip().lower()
        items = payload.get("items")
        if card_type == "family":
            cleaned_items = cls._clean_family_items(items)
        else:
            cleaned_items = cls._clean_quote_or_word_items(items)
        return {"id": card_id, "label": label, "type": card_type, "items": cleaned_items}

    @classmethod
    def _load_registry(cls):
        try:
            payload = cls._read_json(CARDS_PATH) or {}
        except Exception:
            logger.exception("Failed to load card catalog: %s", CARDS_PATH)
            payload = {}

        leaf_meta = {}
        groups = {}
        all_meta = {}
        group_for = {}
        initial_specs = {}

        legacy_cards = payload.get("cards") if isinstance(payload, dict) else None
        registry_entries = payload.get("entries") if isinstance(payload, dict) else None

        # Legacy format: {"cards":[{"id","label","type","items":[...]}, ...]}
        if isinstance(legacy_cards, list) and not isinstance(registry_entries, list):
            for card in legacy_cards:
                spec = cls._clean_card_payload(card)
                if not spec:
                    continue
                meta = {"id": spec["id"], "label": spec["label"], "type": spec["type"], "source": None}
                leaf_meta[meta["id"]] = meta
                all_meta[meta["id"]] = meta
                initial_specs[meta["id"]] = spec
            return {
                "leaf": leaf_meta,
                "groups": groups,
                "all": all_meta,
                "group_for": group_for,
                "legacy": True,
                "initial_specs": initial_specs,
            }

        if not isinstance(registry_entries, list):
            registry_entries = []

        for entry in registry_entries:
            if not isinstance(entry, dict):
                continue
            entry_id = cls._normalize_id(entry.get("id"))
            if not entry_id:
                continue
            entry_type = str(entry.get("type") or "card").strip().lower()
            try:
                source_path = cls._resolve_source(entry.get("source"))
            except Exception:
                logger.exception("Skipping invalid card source for %s", entry_id)
                continue

            if entry_type == "group":
                group_label = str(entry.get("label") or entry_id).strip() or entry_id
                try:
                    group_payload = cls._read_json(source_path) or {}
                except FileNotFoundError:
                    # Optional local registries may not exist; skip silently.
                    continue
                except Exception:
                    logger.exception("Failed to load group registry: %s", source_path)
                    continue

                categories = group_payload.get("categories") if isinstance(group_payload, dict) else None
                if not isinstance(categories, list):
                    categories = []

                children = []
                for child in categories:
                    if not isinstance(child, dict):
                        continue
                    child_id = cls._normalize_id(child.get("id"))
                    if not child_id:
                        continue
                    child_label = str(child.get("label") or child_id).strip() or child_id
                    child_type = str(child.get("type") or "quote").strip().lower()
                    try:
                        child_source = cls._resolve_source(child.get("source"))
                    except Exception:
                        logger.exception("Skipping invalid category source for %s.%s", entry_id, child_id)
                        continue
                    child_meta = {"id": child_id, "label": child_label, "type": child_type, "source": child_source}
                    children.append(child_meta)
                    all_meta[child_id] = child_meta
                    group_for[child_id] = entry_id

                groups[entry_id] = {"id": entry_id, "label": group_label, "children": children}
                continue

            # Leaf card entry (registry points to a standalone json file)
            try:
                card_payload = cls._read_json(source_path) or {}
            except FileNotFoundError:
                # Optional local sources (e.g. family dates) may not exist; skip silently.
                continue
            except Exception:
                logger.exception("Failed to load card definition for %s", entry_id)
                continue

            spec = cls._clean_card_payload(card_payload, fallback_id=entry_id)
            if not spec:
                continue
            meta = {"id": spec["id"], "label": spec["label"], "type": spec["type"], "source": source_path}
            leaf_meta[meta["id"]] = meta
            all_meta[meta["id"]] = meta

        return {
            "leaf": leaf_meta,
            "groups": groups,
            "all": all_meta,
            "group_for": group_for,
            "legacy": False,
            "initial_specs": initial_specs,
        }

    @classmethod
    def _registry(cls):
        cache = getattr(cls, "_registry_cache", None)
        if cache is None:
            cache = cls._load_registry()
            cls._registry_cache = cache
            cls._spec_cache = dict(cache.get("initial_specs") or {})
        return cache

    @classmethod
    def _card_catalog(cls):
        reg = cls._registry()
        return {card_id: {"id": meta["id"], "label": meta["label"], "type": meta["type"]} for card_id, meta in reg["all"].items()}

    @classmethod
    def _card_groups(cls):
        reg = cls._registry()
        groups = list(reg["groups"].values())
        groups.sort(key=lambda g: str(g.get("label") or g.get("id")))
        for group in groups:
            group["children"].sort(key=lambda c: str(c.get("label") or c.get("id")))
        return groups

    @classmethod
    def _card_leaf_choices(cls):
        reg = cls._registry()
        items = list(reg["leaf"].values())
        items.sort(key=lambda m: (m.get("id") != DEFAULT_CARD_ID, str(m.get("label") or m.get("id"))))
        return [{"id": m.get("id"), "label": m.get("label"), "type": m.get("type")} for m in items]

    @classmethod
    def _card_group_for_id(cls, card_id):
        card_id = cls._normalize_id(card_id)
        reg = cls._registry()
        return (reg.get("group_for") or {}).get(card_id, "")

    @classmethod
    def _card_spec(cls, card_id):
        card_id = cls._normalize_id(card_id) or DEFAULT_CARD_ID
        reg = cls._registry()
        meta = reg["all"].get(card_id) or reg["all"].get(DEFAULT_CARD_ID)
        if not meta:
            return {"id": DEFAULT_CARD_ID, "label": DEFAULT_CARD_ID, "type": "quote", "items": []}

        cache = getattr(cls, "_spec_cache", None)
        if cache is None:
            cache = {}
            cls._spec_cache = cache
        if meta["id"] in cache:
            return cache[meta["id"]]

        # Registry leaf/category: load from referenced source json file.
        if meta.get("source"):
            try:
                payload = cls._read_json(meta["source"]) or {}
                cleaned = cls._clean_card_payload(payload, fallback_id=meta["id"])
            except Exception:
                logger.exception("Failed to load card spec: %s", meta.get("source"))
                cleaned = None

            if not cleaned:
                cleaned = {"id": meta["id"], "label": meta["label"], "type": meta["type"], "items": []}
            else:
                cleaned["id"] = meta["id"]
                cleaned["label"] = cleaned.get("label") or meta["label"]
                cleaned["type"] = cleaned.get("type") or meta["type"]

            cache[meta["id"]] = cleaned
            return cleaned

        # Legacy: specs were preloaded into cache (or missing).
        cleaned = {"id": meta["id"], "label": meta["label"], "type": meta["type"], "items": []}
        cache[meta["id"]] = cleaned
        return cleaned

    @staticmethod
    def _format_month_day(value):
        value = (value or "").strip()
        try:
            month, day = [int(p) for p in value.split("-", 1)]
            return datetime(2000, month, day).strftime("%b %d").replace(" 0", " ")
        except Exception:
            return value

    @staticmethod
    def _pick_family_item(items, *, day_key):
        if not items:
            return {"date": "", "text": "No items configured.", "year": None}
        try:
            ref = datetime.strptime(day_key, "%Y-%m-%d").date()
        except Exception:
            ref = datetime.utcnow().date()

        candidates = []
        for item in items:
            if not isinstance(item, dict):
                continue
            md = str(item.get("date") or "").strip()
            if not re.fullmatch(r"\d{2}-\d{2}", md):
                continue
            try:
                month, day = [int(p) for p in md.split("-", 1)]
                when = datetime(ref.year, month, day).date()
                if when < ref:
                    when = datetime(ref.year + 1, month, day).date()
            except Exception:
                continue
            candidates.append((when, item))

        if not candidates:
            return {"date": "", "text": "No items configured.", "year": None}
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def _render_card_left_panel(self, *, card_id, day_key, size, base=None, draw_card_box=False):
        w, h = size
        if base is not None:
            img = base.copy().convert("RGB")
        else:
            img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        spec = self._card_spec(card_id) or {"type": "quote", "items": []}
        card_type = (spec.get("type") or "quote").strip().lower()
        items = spec.get("items") or []

        if card_type == "family":
            pick = self._pick_family_item(items, day_key=day_key)
        else:
            seed = f"{card_id}|{day_key}"
            if items:
                idx = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % len(items)
                pick = items[idx]
            else:
                pick = {"text": "No items configured.", "attribution": ""}

        pad = max(18, int(w * 0.07))
        max_w = w - pad * 2
        card_pad = max(12, int(pad * 0.55))
        content_x = pad + (card_pad if draw_card_box else 0)
        content_max_w = w - (content_x * 2)
        if content_max_w < max(80, int(w * 0.25)):
            content_x = pad
            content_max_w = max_w

        word_font = self._font("Jost", max(18, int(w * 0.12)), bold=True)
        body_size = max(16, int(w * 0.075))
        body_font = self._font("Jost", body_size)
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

        text = (pick.get("text") or "").strip()
        attribution = (pick.get("attribution") or "").strip()

        is_word = card_type == "word"
        is_family = card_type == "family"

        def draw_box(y0, total_h):
            if not draw_card_box:
                return
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

        if is_family:
            md = str(pick.get("date") or "").strip()
            date_label = self._format_month_day(md) if md else ""
            year = pick.get("year")
            year_str = str(year) if isinstance(year, int) else ""

            date_font = self._font("Jost", max(14, int(w * 0.06)))
            name_font = self._font("Jost", max(18, int(w * 0.13)), bold=True)
            year_font = self._font("Jost", max(12, int(w * 0.05)))

            date_lines = wrap(date_label, date_font, content_max_w) if date_label else []
            name_lines = wrap(text, name_font, content_max_w) if text else ["Family"]
            year_lines = wrap(year_str, year_font, content_max_w) if year_str else []

            total_h = 0
            if date_lines:
                total_h += len(date_lines) * (line_height(date_font) + int(max(2, pad * 0.05)))
                total_h += int(pad * 0.25)
            total_h += len(name_lines) * (line_height(name_font) + int(max(2, pad * 0.08)))
            if year_lines:
                total_h += int(pad * 0.25)
                total_h += len(year_lines) * (line_height(year_font) + int(max(2, pad * 0.05)))

            y = max(pad, int((h - total_h) / 2))
            boxed = draw_box(y, total_h)
            if boxed is not None:
                img = boxed
                draw = ImageDraw.Draw(img)
            for line in date_lines:
                draw.text((content_x, y), line, fill=(0, 0, 0), font=date_font)
                y += line_height(date_font) + int(max(2, pad * 0.05))
            if date_lines:
                y += int(pad * 0.25)
            for line in name_lines:
                draw.text((content_x, y), line, fill=(0, 0, 0), font=name_font)
                y += line_height(name_font) + int(max(2, pad * 0.08))
            if year_lines:
                y += int(pad * 0.25)
                for line in year_lines:
                    draw.text((content_x, y), line, fill=(60, 60, 60), font=year_font)
                    y += line_height(year_font) + int(max(2, pad * 0.05))
            return img

        # Fit body font if needed (keep within ~70% height).
        max_body_h = int(h * 0.70)
        for _ in range(8):
            body_lines = wrap(text, body_font, content_max_w)
            body_h = len(body_lines) * line_height(body_font)
            footer_h = (line_height(small_font) * (2 if attribution else 1)) + int(pad * 0.7)
            if body_h + footer_h <= max_body_h or body_size <= 12:
                break
            body_size -= 2
            body_font = self._font("Jost", body_size)

        if is_word:
            word_lines = wrap(text, word_font, content_max_w)
            definition_lines = wrap(attribution, small_font, content_max_w) if attribution else []

            total_h = 0
            total_h += len(word_lines) * (line_height(word_font) + int(max(2, pad * 0.06)))
            if definition_lines:
                total_h += int(pad * 0.35)
                total_h += len(definition_lines) * (line_height(small_font) + int(max(2, pad * 0.04)))

            y = max(pad, int((h - total_h) / 2))
            boxed = draw_box(y, total_h)
            if boxed is not None:
                img = boxed
                draw = ImageDraw.Draw(img)
            for line in word_lines:
                draw.text((content_x, y), line, fill=(0, 0, 0), font=word_font)
                y += line_height(word_font) + int(max(2, pad * 0.06))
            if definition_lines:
                y += int(pad * 0.35)
                for line in definition_lines:
                    draw.text((content_x, y), line, fill=(0, 0, 0), font=small_font)
                    y += line_height(small_font) + int(max(2, pad * 0.04))
        else:
            body_lines = wrap(text, body_font, content_max_w)
            quote_lines = [f"“{body_lines[0]}" if body_lines else "“"]
            quote_lines += body_lines[1:]
            if quote_lines:
                quote_lines[-1] = f"{quote_lines[-1]}”"

            body_h = len(quote_lines) * (line_height(body_font) + int(max(2, pad * 0.08)))
            attr_h = 0
            if attribution:
                attr_h = int(pad * 0.4) + line_height(small_font)
            total_h = body_h + attr_h

            y = max(pad, int((h - total_h) / 2))
            boxed = draw_box(y, total_h)
            if boxed is not None:
                img = boxed
                draw = ImageDraw.Draw(img)
            content_left = content_x
            content_right = w - content_x
            inner_w = max(1, content_right - content_left)
            for line in quote_lines:
                x = content_left + max(0, int((inner_w - text_width(line, body_font)) / 2))
                draw.text((x, y), line, fill=(0, 0, 0), font=body_font)
                y += line_height(body_font) + int(max(2, pad * 0.08))

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
        # Reuse the Daily Cat Weather icon renderer to keep a consistent look.
        from plugins.daily_cat_weather.daily_cat_weather import DailyCatWeather

        return DailyCatWeather._simple_weather_icon(icon_code, size)  # pylint: disable=protected-access
