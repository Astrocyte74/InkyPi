import os
import time
import logging
from datetime import datetime
from typing import Dict
import pytz

from openai import OpenAI
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from model import RefreshInfo
from plugins.ai_image.ai_image import AIImage
from plugins.plugin_registry import get_plugin_instance
from utils.image_utils import compute_image_hash
from utils.app_utils import resolve_path

logger = logging.getLogger(__name__)


class TelegramTextFlow:
    """Manage Telegram `/txt` interactive flows.

    Enhanced to keep custom AI background configuration inline in the
    same Telegram message (single-card flow).
    """

    STYLE_OPTIONS = [
        ("simple", "📝 Simple"),
        ("caption", "🗒️ Caption Box"),
        ("card", "🪧 Quote Card"),
        ("sticky", "📌 Sticky Note"),
    ]

    BACKGROUND_OPTIONS = [
        ("none", "Plain"),
        ("illustration_blur", "Illustration (Blur)"),
        ("illustration", "Illustration"),
        ("ai_image", "Auto-Generate Image"),
        ("color", "Solid Colour"),
        ("saved", "Saved Image"),
        ("custom_ai", "Custom Image (Prompt)"),
    ]

    # Inline AI background options for custom backgrounds
    AI_MODELS = [
        ("dall-e-3", "DALL·E 3"),
        ("gpt-image-1", "GPT Image 1"),
        ("dall-e-2", "DALL·E 2"),
    ]

    QUALITY_OPTIONS = {
        "dall-e-3": ["standard", "hd"],
        "gpt-image-1": ["medium", "high", "low"],
        "dall-e-2": ["standard"],
    }

    PALETTES = [("spectra6", "Colour"), ("bw", "Black & White")]
    STYLE_HINTS = [
        ("none", "🚫 None"),
        ("illustration", "✏️ Illustration"),
        ("drawing", "📝 Drawing"),
        ("far_side", "🐄 Far Side"),
        ("van_gogh", "🖌️ Van Gogh"),
    ]

    def __init__(self, device_config, display_manager, refresh_task, storage_dir):
        self.device_config = device_config
        self.display_manager = display_manager
        self.refresh_task = refresh_task
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

        self.requests: Dict[str, Dict] = {}
        # Weather cache (15 min TTL)
        self._weather_cache = {
            "key": None,
            "ts": 0.0,
            "data": None,  # dict: {temp_text, icon_path, cond_text, provider, units}
        }
        self._WEATHER_TTL_SECONDS = 900

    def _banner_override_today(self):
        try:
            override = self.device_config.get_config("banner_override", default=None)
        except Exception:
            return None
        if not isinstance(override, dict):
            return None
        day = str(override.get("day") or "").strip()
        headline = str(override.get("headline") or "").strip()
        detail = str(override.get("detail") or "").strip()
        if not day or not headline:
            return None
        tz_str = self.device_config.get_config("timezone", default="UTC")
        try:
            tz = pytz.timezone(tz_str)
        except Exception:
            tz = pytz.UTC
        today = datetime.now(tz).date().isoformat()
        if day != today:
            return None
        return {"day": day, "headline": headline, "detail": detail}

    # --- Request lifecycle -------------------------------------------------

    def create_request(self, chat_id, text):
        request_id = f"{chat_id}:{int(time.time() * 1000)}"
        data = {
            "id": request_id,
            "chat_id": chat_id,
            "text": text.strip(),
            "style": "caption",
            "rewrite": False,
            "background": "illustration_blur",
            "message_id": None,
            "locked": False,
            "awaiting_background": False,
            "custom_background": None,
            "image_prompt": text.strip(),
            "awaiting_prompt": False,
            "bg_selected": True,
            # Inline custom background configuration state
            "bg_mode": "summary",  # summary | bg_config
            "bg_model": self.AI_MODELS[0][0],
            "bg_quality": self.QUALITY_OPTIONS[self.AI_MODELS[0][0]][0],
            "bg_palette": self.PALETTES[0][0],
            "bg_style_hint": "none",
            "bg_color_choice": None,
            "awaiting_saved": False,
            "saved_name": None,
            "saved_page": 0,
            "final_text_preview": None,
        }
        self.requests[request_id] = data
        return data

    def get_request(self, request_id):
        return self.requests.get(request_id)

    def set_message_id(self, request_id, message_id):
        request = self.get_request(request_id)
        if request:
            request["message_id"] = message_id

    def cancel_request(self, request_id):
        self.requests.pop(request_id, None)

    # --- Summaries & keyboards --------------------------------------------

    def format_summary(self, request, status=None):
        style_label = dict(self.STYLE_OPTIONS).get(request.get("style"), request.get("style"))
        if not request.get("bg_selected"):
            background_label = "Select one"
        else:
            background_label = dict(self.BACKGROUND_OPTIONS).get(request.get("background"), request.get("background"))
        if request.get("bg_selected") and request.get("background") == "custom_ai":
            if request.get("awaiting_background"):
                background_label = "Custom AI (configure)"
            elif request.get("custom_background"):
                background_label = "Custom AI ✅"
            elif request.get("awaiting_prompt"):
                background_label = "Custom AI (set prompt)"
        rewrite_label = "On" if request.get("rewrite") else "Off"
        # Prefer showing the rewritten text preview if available
        message_text = request.get("final_text_preview") if request.get("final_text_preview") else request.get("text")
        message_header = "Message (rewritten):" if (request.get("final_text_preview") and request.get("rewrite")) else "Message:"
        lines = [
            "📝 Telegram Text",
            "",
            f"{message_header}\n{message_text}",
            "",
            f"Style: {style_label}",
            f"Rewrite: {rewrite_label}",
            f"Background: {background_label}",
        ]
        banner = self._banner_override_today()
        if banner:
            headline = banner.get("headline") or ""
            if len(headline) > 60:
                headline = headline[:57].rstrip() + "…"
            lines.append(f"Banner (today): {headline}")
        if request.get("style") == "simple" and request.get("background") in {
            "illustration",
            "illustration_blur",
            "ai_image",
            "custom_ai",
            "saved",
        }:
            lines.append("Tip: Simple can be hard to read on images — try Caption/Card/Sticky or Solid Colour.")
        if request.get("bg_selected") and request.get("background") == "custom_ai":
            prompt_preview = request.get("image_prompt", "").strip()
            if not prompt_preview:
                prompt_preview = "(empty)"
            if len(prompt_preview) > 60:
                prompt_preview = prompt_preview[:57] + "…"
            lines.append(f"Image prompt: {prompt_preview}")
        if request.get("bg_selected") and request.get("background") == "saved":
            saved_name = (request.get("saved_name") or "").strip()
            if saved_name:
                # Show without internal prefix
                base = self._strip_prefix(saved_name)
                lines.append(f"Selected: {base} ✅")
            else:
                lines.append("Saved image: (none)")
        if status:
            lines.extend(["", status])
        return "\n".join(lines)

    def build_keyboard(self, request):
        request_id = request["id"]

        # Style choice buttons (no cycling)
        def style_btn(value, label):
            active = request.get("style") == value
            text = f"{label} {'✅' if active else ''}".strip()
            return {"text": text, "callback_data": f"txt|{request_id}|style|{value}"}

        style_rows = [
            [
                style_btn("simple", "📝 Simple"),
                style_btn("caption", "🗒️ Caption"),
            ],
            [
                style_btn("card", "🪧 Card"),
                style_btn("sticky", "📌 Sticky"),
            ],
        ]

        # Rewrite toggle buttons
        rewrite_on = request.get("rewrite", False)
        rewrite_row = [
            {"text": f"Rewrite Off {'✅' if not rewrite_on else ''}".strip(), "callback_data": f"txt|{request_id}|rewrite|off"},
            {"text": f"Rewrite On {'✅' if rewrite_on else ''}".strip(), "callback_data": f"txt|{request_id}|rewrite|on"},
        ]

        # Background selection buttons (no cycling)
        def bg_btn(value, label):
            active = bool(request.get("bg_selected")) and request.get("background") == value
            text = f"{label} {'✅' if active else ''}".strip()
            return {"text": text, "callback_data": f"txt|{request_id}|background|{value}"}

        # Split background options across multiple rows for clarity and longer labels
        bg_rows = [
            [
                bg_btn("none", "Plain"),
                bg_btn("illustration_blur", "Illustration (Blur)"),
            ],
            [
                bg_btn("illustration", "Illustration"),
                bg_btn("ai_image", "Auto-Generate Image"),
            ],
            [
                bg_btn("color", "Solid Colour"),
                bg_btn("saved", "Saved Image"),
                bg_btn("custom_ai", "Custom Image (Prompt)"),
            ],
        ]

        keyboard = [
            [{"text": "Choose style:", "callback_data": f"txt|{request_id}|noop"}],
        ]
        keyboard.extend(style_rows)
        keyboard.extend([
            [{"text": "Rewrite:", "callback_data": f"txt|{request_id}|noop"}],
            rewrite_row,
            [{"text": "Banner:", "callback_data": f"txt|{request_id}|noop"}],
            [
                {"text": "📣 Set as Today’s Banner", "callback_data": f"txt|{request_id}|banner|set"},
                {"text": "🧹 Clear Banner", "callback_data": f"txt|{request_id}|banner|clear"},
            ],
            [{"text": "Pick background:", "callback_data": f"txt|{request_id}|noop"}],
        ])
        keyboard.extend(bg_rows)

        # Contextual actions
        if request.get("bg_selected") and request.get("background") == "custom_ai":
            preview = (request.get("image_prompt", "") or "(empty)").strip()
            if len(preview) > 24:
                preview = preview[:21] + "…"
            keyboard.append([
                {"text": f"🖋 Enter Prompt: {preview}", "callback_data": f"txt|{request_id}|set_prompt"},
            ])
            keyboard.append([
                {"text": "✖️ Cancel", "callback_data": f"txt|{request_id}|cancel"},
            ])
        elif request.get("bg_selected") and request.get("background") == "color":
            # Offer a set of colour choices; show Render once a colour is chosen
            colours = [
                ("#141414", "Black"),
                ("#FFFFFF", "White"),
                ("#E60000", "Red"),
                ("#008F11", "Green"),
                ("#0057FF", "Blue"),
                ("#FFD21E", "Yellow"),
            ]
            row = []
            for hex_code, name in colours:
                is_active = request.get("bg_color_choice") == hex_code
                label = f"{name} {'✅' if is_active else ''}".strip()
                row.append({"text": label, "callback_data": f"txt|{request_id}|bg_color|{hex_code}"})
                if len(row) == 3:
                    keyboard.append(row)
                    row = []
            if row:
                keyboard.append(row)
            if request.get("bg_color_choice"):
                keyboard.append([
                    {"text": "🪄 Generate", "callback_data": f"txt|{request_id}|confirm"},
                    {"text": "✖️ Cancel", "callback_data": f"txt|{request_id}|cancel"},
                ])
        elif request.get("bg_selected") and request.get("background") == "saved":
            # Saved image picker with pagination
            names = self._list_saved_names()
            page = int(request.get("saved_page") or 0)
            page_size = 6
            total_pages = max(1, (len(names) + page_size - 1) // page_size)
            if page >= total_pages:
                page = total_pages - 1
                request["saved_page"] = page
            start = page * page_size
            end = start + page_size
            for chunk_start in range(start, min(end, len(names)), 3):
                row = []
                for name in names[chunk_start: min(chunk_start + 3, len(names))]:
                    label = self._strip_prefix(name)
                    if len(label) > 18:
                        label = label[:15] + "…"
                    if request.get("saved_name") == name:
                        label = f"{label} ✅"
                    row.append({"text": label, "callback_data": f"txt|{request_id}|saved_pick|{name}"})
                if row:
                    keyboard.append(row)
            # Pagination controls
            nav = []
            if total_pages > 1 and page > 0:
                nav.append({"text": "◀️ Prev", "callback_data": f"txt|{request_id}|saved_page|{page-1}"})
            if total_pages > 1 and page < total_pages - 1:
                nav.append({"text": "Next ▶️", "callback_data": f"txt|{request_id}|saved_page|{page+1}"})
            if nav:
                keyboard.append(nav)
            # Manage row
            manage = [
                {"text": "🗑 Delete", "callback_data": f"txt|{request_id}|saved_delete"},
                {"text": "✏️ Rename", "callback_data": f"txt|{request_id}|saved_rename"},
                {"text": "🧹 Clear", "callback_data": f"txt|{request_id}|saved_clear"},
            ]
            keyboard.append(manage)
            # Quick preview + Cancel/Generate
            if request.get("saved_name"):
                keyboard.append([
                    {"text": "👁 Preview", "callback_data": f"txt|{request_id}|saved_preview"},
                ])
            if request.get("saved_name"):
                keyboard.append([
                    {"text": "🪄 Generate", "callback_data": f"txt|{request_id}|confirm"},
                    {"text": "✖️ Cancel", "callback_data": f"txt|{request_id}|cancel"},
                ])
            else:
                keyboard.append([
                    {"text": "✖️ Cancel", "callback_data": f"txt|{request_id}|cancel"},
                ])
            # Generate handled by common gating below
        else:
            if request.get("bg_selected"):
                bg = request.get("background")
                ready = False
                if bg in {"none", "ai_image", "illustration", "illustration_blur"}:
                    ready = True
                elif bg == "color" and request.get("bg_color_choice"):
                    ready = True
                elif bg == "saved" and request.get("saved_name"):
                    ready = True
                if ready:
                    keyboard.append([
                        {"text": "🪄 Generate", "callback_data": f"txt|{request_id}|confirm"},
                        {"text": "✖️ Cancel", "callback_data": f"txt|{request_id}|cancel"},
                    ])

        return {"inline_keyboard": keyboard}

    def _list_saved_names(self):
        saved_dir = os.path.join(self.storage_dir, "saved")
        if not os.path.isdir(saved_dir):
            return []
        items = []
        try:
            for fn in os.listdir(saved_dir):
                if fn.lower().endswith(".png"):
                    path = os.path.join(saved_dir, fn)
                    try:
                        mtime = os.path.getmtime(path)
                    except OSError:
                        mtime = 0
                    name = os.path.splitext(fn)[0]
                    items.append((mtime, name))
        except OSError:
            return []
        # Newest first
        items.sort(key=lambda t: t[0], reverse=True)
        return [name for _, name in items]

    # --- Mutators ----------------------------------------------------------

    def cycle_style(self, request):
        keys = [value for value, _ in self.STYLE_OPTIONS]
        current_index = keys.index(request["style"])
        request["style"] = keys[(current_index + 1) % len(keys)]

    def toggle_rewrite(self, request):
        request["rewrite"] = not request.get("rewrite", False)
        request["final_text_preview"] = None

    def cycle_background(self, request):
        keys = [value for value, _ in self.BACKGROUND_OPTIONS]
        current = request.get("background")
        try:
            current_index = keys.index(current)
        except ValueError:
            current_index = 0
        request["background"] = keys[(current_index + 1) % len(keys)]
        request["bg_selected"] = True
        if request["background"] != "custom_ai":
            request["awaiting_background"] = False
            request["custom_background"] = None
            request["awaiting_prompt"] = False
        else:
            if not request.get("image_prompt"):
                request["image_prompt"] = request["text"].strip()
        if request["background"] != "color":
            request["bg_color_choice"] = None
        if request["background"] != "saved":
            request["saved_name"] = None
            request["awaiting_saved"] = False

    # New direct setters (non-cycling)
    def set_style(self, request, style):
        keys = [value for value, _ in self.STYLE_OPTIONS]
        if style in keys:
            request["style"] = style

    def set_rewrite(self, request, enabled: bool):
        request["rewrite"] = bool(enabled)
        request["final_text_preview"] = None

    def set_background(self, request, background):
        aliases = {"latest": "illustration_blur", "weather": "illustration_blur"}
        background = aliases.get(background, background)
        keys = [value for value, _ in self.BACKGROUND_OPTIONS]
        if background in keys:
            request["background"] = background
            request["bg_selected"] = True
            if background != "custom_ai":
                request["awaiting_background"] = False
                request["custom_background"] = None
                request["awaiting_prompt"] = False
            else:
                if not request.get("image_prompt"):
                    request["image_prompt"] = request["text"].strip()
            if background != "color":
                request["bg_color_choice"] = None
            if background != "saved":
                request["saved_name"] = None
                request["awaiting_saved"] = False

    def set_bg_color(self, request, hex_code):
        request["bg_color_choice"] = hex_code

    def set_wbadge(self, request, enabled: bool):
        request["wbadge"] = bool(enabled)

    def set_woverlay(self, request, enabled: bool):
        request["woverlay"] = bool(enabled)

    def await_saved(self, request):
        request["awaiting_saved"] = True

    def set_saved_name(self, request, name):
        request["saved_name"] = (name or "").strip()
        request["awaiting_saved"] = False

    def await_custom_prompt(self, request):
        request["awaiting_prompt"] = True

    def mark_custom_background_pending(self, request):
        request["awaiting_background"] = True
        request["custom_background"] = None
        request["awaiting_prompt"] = False

    # --- Inline background config mutators ---------------------------------

    def enter_bg_config(self, request):
        request["bg_mode"] = "bg_config"

    def exit_bg_config(self, request):
        request["bg_mode"] = "summary"

    def cycle_bg_model(self, request):
        keys = [k for k, _ in self.AI_MODELS]
        idx = keys.index(request.get("bg_model", keys[0]))
        request["bg_model"] = keys[(idx + 1) % len(keys)]
        allowed = self.QUALITY_OPTIONS.get(request["bg_model"], ["standard"])
        if request.get("bg_quality") not in allowed:
            request["bg_quality"] = allowed[0]

    def cycle_bg_quality(self, request):
        allowed = self.QUALITY_OPTIONS.get(request.get("bg_model"), [request.get("bg_quality")])
        idx = allowed.index(request.get("bg_quality", allowed[0]))
        request["bg_quality"] = allowed[(idx + 1) % len(allowed)]

    def cycle_bg_palette(self, request):
        keys = [k for k, _ in self.PALETTES]
        idx = keys.index(request.get("bg_palette", keys[0]))
        request["bg_palette"] = keys[(idx + 1) % len(keys)]

    def cycle_bg_style(self, request):
        keys = [k for k, _ in self.STYLE_HINTS]
        idx = keys.index(request.get("bg_style_hint", keys[0]))
        request["bg_style_hint"] = keys[(idx + 1) % len(keys)]

    def attach_custom_background(self, request_id, background_path):
        request = self.requests.get(request_id)
        if not request:
            return None
        request["custom_background"] = background_path
        request["awaiting_background"] = False
        return request

    def consume_prompt(self, chat_id, text):
        if text.strip().startswith("/") and text.strip().lower() != "/skip":
            return None
        for request in self.requests.values():
            if request["chat_id"] == chat_id and request.get("awaiting_prompt"):
                cleaned = text.strip()
                if cleaned.lower() == "/skip":
                    request["image_prompt"] = request["text"].strip()
                else:
                    request["image_prompt"] = cleaned or request["text"].strip()
                request["awaiting_prompt"] = False
                return request
            if request["chat_id"] == chat_id and request.get("awaiting_saved"):
                cleaned = text.strip()
                request["saved_name"] = cleaned
                request["awaiting_saved"] = False
                return request
            if request["chat_id"] == chat_id and request.get("awaiting_saved_rename"):
                newname = text.strip()
                oldname = request.get("saved_rename_from")
                try:
                    final_name = self.rename_saved(oldname, newname)
                    request["saved_name"] = final_name
                except Exception:
                    logger.exception("Rename saved failed")
                request["awaiting_saved_rename"] = False
                request["saved_rename_from"] = None
                return request
        return None

    # --- Final rendering ---------------------------------------------------

    def finalize(self, request, *, target_size=None):
        # Use precomputed preview when available to keep summary consistent
        final_text = request.get("final_text_preview") or request["text"]
        if request.get("rewrite") and not request.get("final_text_preview"):
            try:
                rewritten = self._rewrite_text(final_text)
                if rewritten:
                    final_text = rewritten
            except Exception as exc:
                logger.exception("Failed to rewrite Telegram text: %s", exc)
                raise RuntimeError("Failed to rewrite text via AI service.")

        if not target_size:
            width, height = self.device_config.get_resolution()
            if self.device_config.get_config("orientation") == "vertical":
                width, height = height, width
            target_size = (width, height)

        background_path = None
        background_color = None
        background_mode = request.get("background")
        placement = "center"
        # Backward compatibility for older keyboards.
        if background_mode == "latest":
            background_mode = "illustration_blur"
        if background_mode == "weather":
            background_mode = "illustration_blur"

        if background_mode == "illustration_blur":
            background_path = self._prepare_illustration_background(size=target_size, blur=True)
            if not background_path:
                logger.warning("No illustration background available; using plain background.")
        elif background_mode == "illustration":
            background_path = self._prepare_illustration_background(size=target_size, blur=False)
            if not background_path:
                logger.warning("No illustration background available; using plain background.")
        elif background_mode == "ai_image":
            background_path = self._generate_ai_background(final_text)
        elif background_mode == "custom_ai":
            background_path = request.get("custom_background")
            if not background_path:
                raise RuntimeError("Custom background not ready yet.")
        elif background_mode == "color":
            background_color = request.get("bg_color_choice")
            if not background_color:
                raise RuntimeError("Choose a colour first.")
        elif background_mode == "saved":
            name = (request.get("saved_name") or "").strip()
            if not name:
                raise RuntimeError("Enter a saved image name.")
            candidate = os.path.join(self.storage_dir, "saved", f"{name}.png")
            if not os.path.exists(candidate):
                logger.warning("Saved image %s not found under telegram/saved", candidate)
                raise RuntimeError("Saved image not found.")
            background_path = candidate
        # If an image background is used, default placement to bottom band
        if background_mode in {"ai_image", "custom_ai", "saved", "illustration", "illustration_blur"}:
            placement = "bottom"

        # Persist last background used for /txt to allow saving later
        try:
            self._persist_last_text_background(background_path, background_color)
        except Exception:
            logger.exception("Failed to persist last text background.")

        image = self._render_text_image(
            final_text,
            request.get("style"),
            background_path,
            background_color,
            placement,
            target_size=target_size,
        )

        return {"image": image, "message": final_text}

    # --- Preview helpers ----------------------------------------------------

    def compute_final_text(self, request):
        """Compute the final text that will be rendered (with rewrite if enabled) without side-effects."""
        text = request.get("text", "")
        if request.get("rewrite"):
            try:
                rewritten = self._rewrite_text(text)
                if rewritten:
                    return rewritten
            except Exception as exc:
                logger.exception("Failed to rewrite Telegram text for preview: %s", exc)
                # Fallback to original text
        return text

    def _persist_last_text_background(self, background_path, background_color):
        dest = os.path.join(self.storage_dir, "last_text_background.png")
        width, height = self.device_config.get_resolution()
        if self.device_config.get_config("orientation") == "vertical":
            width, height = height, width
        if background_color:
            # Create a solid colour background matching device resolution
            img = Image.new("RGB", (width, height), background_color)
            img.save(dest)
        elif background_path and os.path.exists(background_path):
            # Copy by reopening and re-saving to ensure a valid PNG at dest
            with Image.open(background_path) as img:
                bg = img.convert("RGB")
                if bg.size != (width, height):
                    bg = self._cover_crop(bg, (width, height))
                bg.save(dest)

    # --- Weather badge overlay ---------------------------------------------

    def overlay_weather_badge(self, image: Image.Image) -> Image.Image:
        """Public helper to overlay the weather badge onto an image.

        Returns the original image if weather data is unavailable or invalid.
        """
        return self._overlay_weather_badge(image)

    def _overlay_weather_badge(self, image: Image.Image) -> Image.Image:
        data = self._fetch_weather_badge_data()
        if not data:
            return image
        icon_path = data.get("icon_path")
        temp_text = data.get("temp_text", "")
        if not temp_text:
            return image

        img = image.convert("RGBA")
        draw = ImageDraw.Draw(img)
        W, H = img.size
        # Badge sizing
        font_path = resolve_path("static/fonts/Jost-SemiBold.ttf")
        font_size = max(18, int(min(W, H) * 0.05))
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        padding = int(font_size * 0.5)
        icon_size = int(font_size * 1.4)
        # Measure text bbox
        bbox = draw.textbbox((0, 0), temp_text, font=font)
        tw = max(0, bbox[2] - bbox[0])
        th = max(0, bbox[3] - bbox[1])
        bw = padding * 3 + icon_size + tw
        bh = padding * 2 + max(icon_size, th)

        # Position from global Telegram weather options (default top-right)
        pos = (((self._get_telegram_weather_options().get("weather") or {}).get("badge") or {}).get("position") or "tr").lower()
        if pos not in {"tr", "tl", "br", "bl"}:
            pos = "tr"
        if pos == "tr":
            x1 = max(0, W - padding - bw)
            y1 = padding
        elif pos == "tl":
            x1 = padding
            y1 = padding
        elif pos == "br":
            x1 = max(0, W - padding - bw)
            y1 = max(0, H - padding - bh)
        else:  # bl
            x1 = padding
            y1 = max(0, H - padding - bh)
        x2 = x1 + bw
        y2 = y1 + bh
        # Background rounded rectangle (semi-opaque)
        radius = int(font_size * 0.5)
        try:
            draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=(0, 0, 0, 200))
        except Exception:
            draw.rectangle((x1, y1, x2, y2), fill=(0, 0, 0, 200))

        # Icon
        if icon_path and os.path.exists(icon_path):
            try:
                with Image.open(icon_path) as ic:
                    ic = ic.convert("RGBA").resize((icon_size, icon_size))
                    img.paste(ic, (x1 + padding, y1 + (bh - icon_size) // 2), ic)
            except Exception:
                logger.exception("Failed to draw weather icon")
        # Text (white)
        tx = x1 + padding * 2 + icon_size
        ty = y1 + (bh - th) // 2
        draw.text((tx, ty), temp_text, font=font, fill=(255, 255, 255, 255))

        return img.convert("RGB")

    def _fetch_weather_badge_data(self):
        # Use cached fetch to avoid excessive API calls
        info = self._get_cached_weather()
        if not info:
            return None
        return {"icon_path": info.get("icon_path"), "temp_text": info.get("temp_text")}

    def _get_cached_weather(self):
        plugin, settings = self._get_weather_plugin_and_settings()
        if not plugin or not settings:
            return None
        provider = (settings.get("weatherProvider") or "OpenWeatherMap").strip()
        units = (settings.get("units") or "metric").strip()
        # Validate coordinates early to avoid API errors
        def _pf(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        lat = _pf(settings.get("latitude"))
        lon = _pf(settings.get("longitude"))
        if lat is None or lon is None:
            return None

        key = f"{provider}|{units}|{lat:.4f}|{lon:.4f}"
        now = time.time()
        if (
            self._weather_cache.get("key") == key
            and (now - float(self._weather_cache.get("ts", 0))) < self._WEATHER_TTL_SECONDS
            and self._weather_cache.get("data")
        ):
            return self._weather_cache["data"]

        try:
            if provider == "OpenWeatherMap":
                api_key = self.device_config.load_env_key("OPEN_WEATHER_MAP_SECRET")
                if not api_key:
                    return None
                wd = plugin.get_weather_data(api_key, units, lat, lon)
                current = wd.get("current", {})
                temp = current.get("temp")
                wobj = (current.get("weather") or [{}])[0]
                icon_code = (wobj.get("icon", "01d") or "01d").replace("n", "d")
                cond_text = (wobj.get("description") or wobj.get("main") or "").strip()
                icon_path = plugin.get_plugin_dir(f"icons/{icon_code}.png")
            elif provider == "OpenMeteo":
                wd = plugin.get_open_meteo_data(lat, lon, units, 1)
                current = wd.get("current_weather", {})
                temp = current.get("temperature")
                from datetime import datetime as _dt
                from datetime import timezone as _tz
                hour = _dt.now(_tz.utc).hour
                icon_code = plugin.map_weather_code_to_icon(current.get("weathercode", 0), hour)
                icon_path = plugin.get_plugin_dir(f"icons/{icon_code}.png")
                # Minimal condition mapping for Open‑Meteo (WMO codes)
                wmo = int(current.get("weathercode", 0) or 0)
                cond_map = {
                    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                    45: "Fog", 48: "Fog",
                    51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
                    56: "Freezing drizzle", 57: "Freezing drizzle",
                    61: "Rain", 63: "Rain", 65: "Heavy rain",
                    66: "Freezing rain", 67: "Freezing rain",
                    71: "Snow", 73: "Snow", 75: "Heavy snow",
                    77: "Snow grains",
                    80: "Rain showers", 81: "Rain showers", 82: "Heavy showers",
                    85: "Snow showers", 86: "Snow showers",
                    95: "Thunderstorm", 96: "Thunder + hail", 99: "Thunder + hail",
                }
                cond_text = cond_map.get(wmo, "")
            else:
                return None
            unit_symbol = {"metric": "°C", "imperial": "°F", "standard": "K"}.get(units, "°C")
            temp_text = f"{int(round(temp))}{unit_symbol}" if isinstance(temp, (int, float)) else None
            data = {
                "icon_path": icon_path,
                "temp_text": temp_text,
                "cond_text": cond_text,
                "provider": provider,
                "units": units,
                "lat": lat,
                "lon": lon,
                "ts": now,
            }
            self._weather_cache.update({"key": key, "ts": now, "data": data})
            return data
        except Exception:
            logger.exception("Failed to fetch weather data (cached)")
            return None

    # --- Saved image helpers -----------------------------------------------

    def _list_saved_names(self):
        saved_dir = os.path.join(self.storage_dir, "saved")
        if not os.path.isdir(saved_dir):
            return []
        items = []
        try:
            for fn in os.listdir(saved_dir):
                if fn.lower().endswith(".png"):
                    path = os.path.join(saved_dir, fn)
                    try:
                        mtime = os.path.getmtime(path)
                    except OSError:
                        mtime = 0
                    name = os.path.splitext(fn)[0]
                    items.append((mtime, name))
        except OSError:
            return []
        # Newest first
        items.sort(key=lambda t: t[0], reverse=True)
        return [name for _, name in items]

    def _sanitize_name(self, name):
        cleaned = "".join(c if c.isalnum() or c in {"-", "_"} else "-" for c in (name or "").strip())
        while "--" in cleaned:
            cleaned = cleaned.replace("--", "-")
        cleaned = cleaned.strip("-")
        if not cleaned:
            raise ValueError("Invalid name.")
        return cleaned.lower()

    def _strip_prefix(self, name):
        for prefix in ("composite_", "txtbg_", "bg_"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def delete_saved(self, name):
        saved_dir = os.path.join(self.storage_dir, "saved")
        path = os.path.join(saved_dir, f"{name}.png")
        if os.path.exists(path):
            os.remove(path)
        else:
            raise FileNotFoundError("Saved image not found.")

    def rename_saved(self, old, new):
        saved_dir = os.path.join(self.storage_dir, "saved")
        old_path = os.path.join(saved_dir, f"{old}.png")
        if not os.path.exists(old_path):
            raise FileNotFoundError("Original saved image not found.")
        # Preserve the original prefix; user-entered name updates only the base
        prefix = ""
        for p in ("composite_", "txtbg_", "bg_"):
            if old.startswith(p):
                prefix = p
                break
        base = self._sanitize_name(self._strip_prefix(new))
        safe = f"{prefix}{base}" if prefix else base
        new_path = os.path.join(saved_dir, f"{safe}.png")
        if os.path.exists(new_path):
            # Deduplicate with numeric suffix on base, keep prefix
            i = 1
            base0 = base
            while os.path.exists(new_path):
                base = f"{base0}-{i}"
                safe = f"{prefix}{base}" if prefix else base
                new_path = os.path.join(saved_dir, f"{safe}.png")
                i += 1
        os.rename(old_path, new_path)
        return safe

    # --- Helpers -----------------------------------------------------------

    def _rewrite_text(self, text):
        api_key = self.device_config.load_env_key("OPEN_AI_SECRET")
        if not api_key:
            raise RuntimeError("OPEN AI API Key not configured.")

        ai_plugin = self._get_ai_plugin()
        if not ai_plugin:
            raise RuntimeError("AI Image plugin is required for prompt service.")

        client = OpenAI(api_key=api_key)
        prompt_client = ai_plugin._get_prompt_client(self.device_config, client)  # pylint: disable=protected-access

        system_content = (
            "You polish short notes for an e-ink display. Keep the meaning, limit to 35 words, "
            "and favour concise, readable phrasing."
        )
        user_content = text.strip()
        rewritten = AIImage._call_prompt_service(prompt_client, system_content, user_content, temperature=0.5)  # pylint: disable=protected-access
        logger.info("Rewrote Telegram text: %s -> %s", text, rewritten)
        return rewritten.strip()

    def _generate_ai_background(self, prompt_text, bg_model=None, bg_quality=None, bg_palette=None, style_hint=None):
        ai_plugin = self._get_ai_plugin()
        if not ai_plugin:
            raise RuntimeError("AI Image plugin is required to generate backgrounds.")

        api_key = self.device_config.load_env_key("OPEN_AI_SECRET")
        if not api_key:
            raise RuntimeError("OPEN AI API Key not configured.")

        model = (bg_model or "dall-e-3").strip()
        quality = (bg_quality or (self.QUALITY_OPTIONS.get(model) or ["standard"])[0]).strip()
        palette = (bg_palette or "spectra6").strip().lower()

        settings = {
            "textPrompt": prompt_text,
            "imageModel": model,
            "quality": quality,
            "palette": palette,
        }
        if style_hint and style_hint != "none":
            settings["styleHint"] = style_hint
        image = ai_plugin.generate_image(settings, self.device_config)
        filename = datetime.utcnow().strftime("telegram_text_bg_%Y%m%d_%H%M%S.png")
        path = os.path.join(self.storage_dir, filename)
        image.save(path)
        logger.info("Generated AI background for Telegram text at %s", path)
        return path

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

    def _prepare_illustration_background(self, *, size, cache_id="", blur=False):
        path = self._find_latest_cat_bg(self.device_config, cache_id=cache_id)
        if not path:
            return ""

        try:
            with Image.open(path) as img:
                bg = img.convert("RGB")
        except Exception:
            logger.exception("Failed to load illustration background: %s", path)
            return ""

        if bg.size != size:
            bg = self._cover_crop(bg, size)

        if blur:
            radius = max(2, min(14, int(size[0] * 0.018)))
            bg = bg.filter(ImageFilter.GaussianBlur(radius=radius))
            bg = ImageEnhance.Brightness(bg).enhance(0.72)
            bg = ImageEnhance.Color(bg).enhance(0.60)
        else:
            bg = ImageEnhance.Brightness(bg).enhance(0.86)
            bg = ImageEnhance.Color(bg).enhance(0.85)

        safe_cache = "".join(c for c in (cache_id or "auto") if c.isalnum() or c in {"-", "_"}).strip("-_")
        safe_cache = safe_cache or "auto"
        mode = "blur" if blur else "plain"
        filename = f"telegram_text_ill_bg_{safe_cache}_{size[0]}x{size[1]}_{mode}.png"
        out_path = os.path.join(self.storage_dir, filename)
        try:
            bg.save(out_path)
        except Exception:
            logger.exception("Failed to write illustration background: %s", out_path)
            return ""
        return out_path

    def _generate_weather_background(self):
        plugin, settings = self._get_weather_plugin_and_settings()
        if not plugin:
            raise RuntimeError("Weather plugin is not installed.")
        if not settings:
            raise RuntimeError("Weather plugin is not configured.")
        image = plugin.generate_image(settings, self.device_config)
        filename = datetime.utcnow().strftime("telegram_text_weather_bg_%Y%m%d_%H%M%S.png")
        path = os.path.join(self.storage_dir, filename)
        image.save(path)
        logger.info("Generated Weather background for Telegram text at %s", path)
        return path

    def _render_text_image(self, text, style, background_path, background_color=None, placement="center", target_size=None):
        plugin = self._get_text_plugin()
        if not plugin:
            raise RuntimeError("Telegram Text plugin is not registered.")

        settings = {
            "text": text,
            "style": style,
            "background_path": background_path,
        }
        if target_size:
            settings["target_size"] = target_size
        if background_color:
            settings["background_color"] = background_color
        if placement:
            settings["placement"] = placement
        return plugin.generate_image(settings, self.device_config)

    def _save_image(self, image):
        filename = datetime.utcnow().strftime("telegram_text_%Y%m%d_%H%M%S.png")
        path = os.path.join(self.storage_dir, filename)
        image.save(path)

        latest_path = os.path.join(self.storage_dir, "latest_text.png")
        image.save(latest_path)
        return path

    def _display_image(self, image, final_text):
        current_dt = (
            self.refresh_task._get_current_datetime()
            if hasattr(self.refresh_task, "_get_current_datetime")
            else datetime.utcnow()
        )
        image_hash = compute_image_hash(image)
        self.display_manager.display_image(image)

        refresh_info = RefreshInfo(
            refresh_type="Telegram Text",
            plugin_id="telegram_text",
            refresh_time=current_dt.isoformat(),
            image_hash=image_hash,
        )
        self.device_config.refresh_info = refresh_info
        self.device_config.write_config()

    def _get_ai_plugin(self):
        plugin_config = self.device_config.get_plugin("ai_image")
        if not plugin_config:
            return None
        return get_plugin_instance(plugin_config)

    def _get_text_plugin(self):
        plugin_config = self.device_config.get_plugin("telegram_text")
        if not plugin_config:
            return None
        return get_plugin_instance(plugin_config)

    def _get_weather_plugin_and_settings(self):
        """Return (plugin, settings) for the first configured Weather plugin instance.

        Preference order:
        - Active playlist (if known), else determined by current time
        - Fallback to the first playlist containing a weather instance
        Returns (None, None) if not available.
        """
        try:
            pm = self.device_config.get_playlist_manager()
            # Determine an active playlist
            current_dt = (
                self.refresh_task._get_current_datetime()
                if hasattr(self.refresh_task, "_get_current_datetime")
                else datetime.utcnow()
            )
            playlist = None
            if pm.active_playlist:
                playlist = pm.get_playlist(pm.active_playlist)
            if not playlist:
                playlist = pm.determine_active_playlist(current_dt)
            # Fallback: first playlist with weather
            candidate_playlists = []
            if playlist:
                candidate_playlists.append(playlist)
            for p in pm.playlists:
                if p is playlist:
                    continue
                candidate_playlists.append(p)

            for p in candidate_playlists:
                for inst in p.plugins:
                    if getattr(inst, "plugin_id", None) == "weather":
                        plugin_config = self.device_config.get_plugin("weather")
                        if not plugin_config:
                            return None, None
                        plugin = get_plugin_instance(plugin_config)
                        return plugin, getattr(inst, "settings", {})
        except Exception:
            logger.exception("Failed to resolve weather plugin instance/settings")
            return None, None
        return None, None

    def _get_telegram_weather_options(self):
        """Return telegram options dict (weather defaults) from device config, with safe defaults."""
        cfg = self.device_config.get_config("telegram_options", default={}) or {}
        return cfg

    def overlay_weather_caption(self, image: Image.Image) -> Image.Image:
        """Overlay a full-width bottom caption with current weather summary.

        Uses same font family as text caption and semi-opaque band.
        Returns the original image if weather data is not available.
        """
        # Reuse badge data for temperature and icon; add basic condition text for OWM
        plugin, settings = self._get_weather_plugin_and_settings()
        if not plugin or not settings:
            return image
        data = self._fetch_weather_badge_data()
        if not data:
            return image
        temp_text = data.get("temp_text") or ""
        icon_path = data.get("icon_path")

        # Use cached data to include condition text without extra fetch
        info = self._get_cached_weather() or {}
        cond_text = info.get("cond_text")
        summary_parts = [p for p in [temp_text, (cond_text.title() if isinstance(cond_text, str) and cond_text else None)] if p]
        summary = " • ".join(summary_parts) if summary_parts else temp_text
        if not summary:
            return image

        img = image.convert("RGBA")
        draw = ImageDraw.Draw(img)
        W, H = img.size
        font_path = resolve_path("static/fonts/Jost-SemiBold.ttf")
        # Initial font size relative to height
        font_size = max(20, int(H * 0.08))
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        padding_x = int(font_size * 0.6)
        padding_y = int(font_size * 0.6)
        icon_size = int(font_size * 1.2)

        # Reduce font size until the text fits width minus icon + paddings
        def measure(fs):
            try:
                f = ImageFont.truetype(font_path, fs)
            except Exception:
                f = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), summary, font=f)
            return f, max(0, bbox[2]-bbox[0]), max(0, bbox[3]-bbox[1])

        f = font
        tw, th = draw.textbbox((0,0), summary, font=f)[2:4]
        available = W - padding_x*3 - icon_size
        while (tw > available or th > int(H*0.25)) and font_size > 12:
            font_size -= 2
            f, tw, th = measure(font_size)
        font = f
        bw = W  # full width band
        bh = th + padding_y*2
        x1 = 0
        y1 = H - bh
        x2 = W
        y2 = H
        # Background band
        try:
            draw.rectangle((x1, y1, x2, y2), fill=(0,0,0,200))
        except Exception:
            draw.rectangle((x1, y1, x2, y2), fill=(0,0,0,200))

        # Icon
        if icon_path and os.path.exists(icon_path):
            try:
                with Image.open(icon_path) as ic:
                    ic = ic.convert("RGBA").resize((icon_size, icon_size))
                    icon_y = y1 + (bh - icon_size)//2
                    img.paste(ic, (x1 + padding_x, icon_y), ic)
            except Exception:
                logger.exception("Failed to draw weather overlay icon")

        # Text
        tx = x1 + padding_x*2 + icon_size
        ty = y1 + (bh - th)//2
        draw.text((tx, ty), summary, font=font, fill=(255,255,255,255))

        # Last updated time (right aligned)
        try:
            info = self._get_cached_weather() or {}
            ts = float(info.get("ts") or time.time())
            tz_str = self.device_config.get_config("timezone", default="UTC")
            time_fmt = self.device_config.get_config("time_format", default="12h")
            dt = datetime.fromtimestamp(ts, tz=pytz.timezone(tz_str))
            if time_fmt == "24h":
                updated = dt.strftime("Updated %H:%M")
            else:
                updated = dt.strftime("Updated %I:%M %p").replace("Updated 0", "Updated ")
            meta_font_size = max(12, int(font_size * 0.6))
            try:
                meta_font = ImageFont.truetype(font_path, meta_font_size)
            except Exception:
                meta_font = ImageFont.load_default()
            mb = draw.textbbox((0,0), updated, font=meta_font)
            mtw = max(0, mb[2]-mb[0])
            mth = max(0, mb[3]-mb[1])
            rx = max(x1 + padding_x, x2 - padding_x - mtw)
            ry = y1 + (bh - mth)//2
            draw.text((rx, ry), updated, font=meta_font, fill=(255,255,255,220))
        except Exception:
            logger.exception("Failed to draw weather updated time")

        return img.convert("RGB")
