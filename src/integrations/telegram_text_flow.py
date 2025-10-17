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
    """Manage Telegram `/txt` interactive flows."""

    STYLE_OPTIONS = [
        ("simple", "📝 Simple"),
        ("caption", "🗒️ Caption Box"),
        ("sticky", "📌 Sticky Note"),
    ]

    BACKGROUND_OPTIONS = [
        ("none", "None"),
        ("latest", "Latest Display"),
        ("ai_image", "AI Image"),
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
        if status:
            lines.extend(["", status])
        return "\n".join(lines)

    def build_keyboard(self, request):
        request_id = request["id"]
        style_label = dict(self.STYLE_OPTIONS).get(request["style"], request["style"])
        rewrite_label = "On" if request["rewrite"] else "Off"
        background_label = dict(self.BACKGROUND_OPTIONS).get(request["background"], request["background"])

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
            [
                {
                    "text": "✅ Send",
                    "callback_data": f"txt|{request_id}|confirm",
                },
                {
                    "text": "✖️ Cancel",
                    "callback_data": f"txt|{request_id}|cancel",
                },
            ],
        ]
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

    def _generate_ai_background(self, prompt_text):
        ai_plugin = self._get_ai_plugin()
        if not ai_plugin:
            raise RuntimeError("AI Image plugin is required to generate backgrounds.")

        api_key = self.device_config.load_env_key("OPEN_AI_SECRET")
        if not api_key:
            raise RuntimeError("OPEN AI API Key not configured.")

        settings = {
            "textPrompt": prompt_text,
            "imageModel": "dall-e-3",
            "quality": "standard",
            "palette": "spectra6",
        }
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
