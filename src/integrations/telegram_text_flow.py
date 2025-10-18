import os
import time
import logging
from datetime import datetime
from typing import Dict

from openai import OpenAI

from model import RefreshInfo
from plugins.ai_image.ai_image import AIImage
from plugins.plugin_registry import get_plugin_instance
from utils.image_utils import compute_image_hash

logger = logging.getLogger(__name__)


class TelegramTextFlow:
    """Manage Telegram `/txt` interactive flows.

    Enhanced to keep custom AI background configuration inline in the
    same Telegram message (single-card flow).
    """

    STYLE_OPTIONS = [
        ("simple", "📝 Simple"),
        ("caption", "🗒️ Caption Box"),
        ("sticky", "📌 Sticky Note"),
    ]

    BACKGROUND_OPTIONS = [
        ("none", "None"),
        ("latest", "Use Last Image"),
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

    # --- Request lifecycle -------------------------------------------------

    def create_request(self, chat_id, text):
        request_id = f"{chat_id}:{int(time.time() * 1000)}"
        data = {
            "id": request_id,
            "chat_id": chat_id,
            "text": text.strip(),
            "style": "caption",
            "rewrite": False,
            "background": "none",
            "message_id": None,
            "locked": False,
            "awaiting_background": False,
            "custom_background": None,
            "image_prompt": text.strip(),
            "awaiting_prompt": False,
            "bg_selected": False,
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
        if request.get("bg_selected") and request.get("background") == "custom_ai":
            prompt_preview = request.get("image_prompt", "").strip()
            if not prompt_preview:
                prompt_preview = "(empty)"
            if len(prompt_preview) > 60:
                prompt_preview = prompt_preview[:57] + "…"
            lines.append(f"Image prompt: {prompt_preview}")
        if request.get("bg_selected") and request.get("background") == "saved":
            saved_name = (request.get("saved_name") or "").strip() or "(none)"
            lines.append(f"Saved image: {saved_name}")
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

        style_row = [
            style_btn("simple", "📝 Simple"),
            style_btn("caption", "🗒️ Caption"),
            style_btn("sticky", "📌 Sticky"),
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

        # Split background options across three rows for clarity and longer labels
        bg_rows = [
            [
                bg_btn("none", "None"),
                bg_btn("ai_image", "Auto-Generate Image"),
            ],
            [
                bg_btn("latest", "Use Last Image"),
                bg_btn("saved", "Saved Image"),
            ],
            [
                bg_btn("color", "Solid Colour"),
                bg_btn("custom_ai", "Custom Image (Prompt)"),
            ],
        ]

        # Labels above each section for clarity
        keyboard = [
            [{"text": "Choose style:", "callback_data": f"txt|{request_id}|noop"}],
            style_row,
            [{"text": "Rewrite:", "callback_data": f"txt|{request_id}|noop"}],
            rewrite_row,
            [{"text": "Pick background:", "callback_data": f"txt|{request_id}|noop"}],
        ]
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
                    label = name
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
            # Manual entry fallback
            keyboard.append([
                {"text": "Enter Name…", "callback_data": f"txt|{request_id}|saved_enter"},
                {"text": "✖️ Cancel", "callback_data": f"txt|{request_id}|cancel"},
            ])
            # Generate handled by common gating below
        else:
            if request.get("bg_selected"):
                bg = request.get("background")
                ready = False
                if bg in {"none", "ai_image", "latest"}:
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
        current_index = keys.index(request["background"])
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
        return None

    # --- Final rendering ---------------------------------------------------

    def finalize(self, request):
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

        background_path = None
        background_color = None
        background_mode = request.get("background")
        placement = "center"
        if background_mode == "latest":
            # Prefer the latest base image saved by Telegram (non-text),
            # fall back to the displayed image if not available.
            telegram_latest = os.path.join(self.storage_dir, "latest.png")
            candidate = telegram_latest if os.path.exists(telegram_latest) else self.device_config.current_image_file
            if candidate and os.path.exists(candidate):
                background_path = candidate
            else:
                logger.warning("Latest display image not found; using solid background.")
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
        if background_mode in {"ai_image", "custom_ai", "latest"}:
            placement = "bottom"

        image = self._render_text_image(final_text, request.get("style"), background_path, background_color, placement)
        saved_path = self._save_image(image)
        self._display_image(image, final_text)

        return {
            "image_path": saved_path,
            "message": final_text,
        }

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

    def _render_text_image(self, text, style, background_path, background_color=None, placement="center"):
        plugin = self._get_text_plugin()
        if not plugin:
            raise RuntimeError("Telegram Text plugin is not registered.")

        settings = {
            "text": text,
            "style": style,
            "background_path": background_path,
        }
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
