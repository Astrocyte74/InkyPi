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
        ("latest", "Latest Display"),
        ("ai_image", "Auto AI Image"),
        ("custom_ai", "Custom AI Image…"),
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
            # Inline custom background configuration state
            "bg_mode": "summary",  # summary | bg_config
            "bg_model": self.AI_MODELS[0][0],
            "bg_quality": self.QUALITY_OPTIONS[self.AI_MODELS[0][0]][0],
            "bg_palette": self.PALETTES[0][0],
            "bg_style_hint": "none",
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
        background_label = dict(self.BACKGROUND_OPTIONS).get(request.get("background"), request.get("background"))
        if request.get("background") == "custom_ai":
            if request.get("awaiting_background"):
                background_label = "Custom AI (configure)"
            elif request.get("custom_background"):
                background_label = "Custom AI ✅"
            elif request.get("awaiting_prompt"):
                background_label = "Custom AI (set prompt)"
        rewrite_label = "On" if request.get("rewrite") else "Off"
        lines = [
            "📝 Telegram Text",
            "",
            f"Message:\n{request['text']}",
            "",
            f"Style: {style_label}",
            f"Rewrite: {rewrite_label}",
            f"Background: {background_label}",
        ]
        if request.get("background") == "custom_ai":
            prompt_preview = request.get("image_prompt", "").strip()
            if not prompt_preview:
                prompt_preview = "(empty)"
            if len(prompt_preview) > 60:
                prompt_preview = prompt_preview[:57] + "…"
            lines.append(f"Image prompt: {prompt_preview}")
        if status:
            lines.extend(["", status])
        return "\n".join(lines)

    def build_keyboard(self, request):
        request_id = request["id"]
        style_label = dict(self.STYLE_OPTIONS).get(request["style"], request["style"])
        rewrite_label = "On" if request["rewrite"] else "Off"
        background_label = dict(self.BACKGROUND_OPTIONS).get(request["background"], request["background"])

        if request.get("background") == "custom_ai":
            if request.get("awaiting_background"):
                background_label = "Custom AI (configure)"
            elif request.get("custom_background"):
                background_label = "Custom AI ✅"

        if request.get("awaiting_background"):
            return {
                "inline_keyboard": [
                    [
                        {
                            "text": "✖️ Cancel",
                            "callback_data": f"txt|{request_id}|cancel",
                        }
                    ]
                ]
            }

        # Background config subview for custom AI
        if request.get("background") == "custom_ai" and request.get("bg_mode") == "bg_config" and not request.get("locked"):
            model_label = dict(self.AI_MODELS).get(request.get("bg_model"), request.get("bg_model"))
            quality_label = (request.get("bg_quality") or "").capitalize()
            palette_label = dict(self.PALETTES).get(request.get("bg_palette"), request.get("bg_palette"))
            style_hint_label = dict(self.STYLE_HINTS).get(request.get("bg_style_hint"), request.get("bg_style_hint"))

            keyboard = [
                [
                    {"text": f"⚙️ Model: {model_label}", "callback_data": f"txt|{request_id}|bg_model"},
                    {"text": f"📐 Quality: {quality_label}", "callback_data": f"txt|{request_id}|bg_quality"},
                ],
                [
                    {"text": f"🎨 Palette: {palette_label}", "callback_data": f"txt|{request_id}|bg_palette"},
                    {"text": f"🧭 Style: {style_hint_label}", "callback_data": f"txt|{request_id}|bg_style"},
                ],
            ]

            preview = (request.get("image_prompt", "") or "(empty)").strip()
            if len(preview) > 20:
                preview = preview[:17] + "…"
            keyboard.append(
                [
                    {"text": f"🖋 Prompt: {preview}", "callback_data": f"txt|{request_id}|set_prompt"},
                ]
            )

            keyboard.append(
                [
                    {"text": "🚀 Generate", "callback_data": f"txt|{request_id}|confirm"},
                    {"text": "⬅️ Back", "callback_data": f"txt|{request_id}|bg_back"},
                ]
            )
            return {"inline_keyboard": keyboard}

        keyboard = [
            [
                {
                    "text": f"🎨 Style: {style_label}",
                    "callback_data": f"txt|{request_id}|cycle_style",
                }
            ],
            [
                {
                    "text": f"✍️ Rewrite: {rewrite_label}",
                    "callback_data": f"txt|{request_id}|toggle_rewrite",
                }
            ],
            [
                {
                    "text": f"🖼 Background: {background_label}",
                    "callback_data": f"txt|{request_id}|cycle_background",
                }
            ],
        ]

        if request.get("background") == "custom_ai":
            preview = request.get("image_prompt", "").strip() or "(empty)"
            if len(preview) > 20:
                preview = preview[:17] + "…"
            keyboard.append(
                [
                    {
                        "text": f"🖋 Prompt: {preview}",
                        "callback_data": f"txt|{request_id}|set_prompt",
                    }
                ]
            )

            keyboard.append(
                [
                    {
                        "text": "⚙️ Background Options",
                        "callback_data": f"txt|{request_id}|bg_config",
                    }
                ]
            )

        keyboard.append(
            [
                {
                    "text": "✅ Send",
                    "callback_data": f"txt|{request_id}|confirm",
                },
                {
                    "text": "✖️ Cancel",
                    "callback_data": f"txt|{request_id}|cancel",
                },
            ]
        )
        return {"inline_keyboard": keyboard}

    # --- Mutators ----------------------------------------------------------

    def cycle_style(self, request):
        keys = [value for value, _ in self.STYLE_OPTIONS]
        current_index = keys.index(request["style"])
        request["style"] = keys[(current_index + 1) % len(keys)]

    def toggle_rewrite(self, request):
        request["rewrite"] = not request.get("rewrite", False)

    def cycle_background(self, request):
        keys = [value for value, _ in self.BACKGROUND_OPTIONS]
        current_index = keys.index(request["background"])
        request["background"] = keys[(current_index + 1) % len(keys)]
        if request["background"] != "custom_ai":
            request["awaiting_background"] = False
            request["custom_background"] = None
            request["awaiting_prompt"] = False
        else:
            request["bg_mode"] = "summary"
            if not request.get("image_prompt"):
                request["image_prompt"] = request["text"].strip()

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
        return None

    # --- Final rendering ---------------------------------------------------

    def finalize(self, request):
        final_text = request["text"]
        if request.get("rewrite"):
            try:
                rewritten = self._rewrite_text(final_text)
                if rewritten:
                    final_text = rewritten
            except Exception as exc:
                logger.exception("Failed to rewrite Telegram text: %s", exc)
                raise RuntimeError("Failed to rewrite text via AI service.")

        background_path = None
        background_mode = request.get("background")
        if background_mode == "latest":
            candidate = self.device_config.current_image_file
            if candidate and os.path.exists(candidate):
                background_path = candidate
            else:
                logger.warning("Latest display image not found; using solid background.")
        elif background_mode == "ai_image":
            background_path = self._generate_ai_background(final_text)
        elif background_mode == "custom_ai":
            background_path = request.get("custom_background")
            if not background_path:
                background_path = self._generate_ai_background(
                    request.get("image_prompt") or final_text,
                    bg_model=request.get("bg_model"),
                    bg_quality=request.get("bg_quality"),
                    bg_palette=request.get("bg_palette"),
                    style_hint=request.get("bg_style_hint"),
                )

        image = self._render_text_image(final_text, request.get("style"), background_path)
        saved_path = self._save_image(image)
        self._display_image(image, final_text)

        return {
            "image_path": saved_path,
            "message": final_text,
        }

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

    def _render_text_image(self, text, style, background_path):
        plugin = self._get_text_plugin()
        if not plugin:
            raise RuntimeError("Telegram Text plugin is not registered.")

        settings = {
            "text": text,
            "style": style,
            "background_path": background_path,
        }
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
