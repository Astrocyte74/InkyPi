import json
import os
import threading
import time
import logging
from datetime import datetime
from io import BytesIO

import requests
from PIL import Image

from model import RefreshInfo
from utils.image_utils import compute_image_hash
from plugins.plugin_registry import get_plugin_instance

logger = logging.getLogger(__name__)


class TelegramBotListener:
    """Background poller that listens for Telegram messages and triggers display updates."""

    API_BASE = "https://api.telegram.org/bot{token}"
    FILE_BASE = "https://api.telegram.org/file/bot{token}"

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

    STYLE_OPTIONS = [
        ("none", "🚫 None"),
        ("randomize", "🎲 Randomize"),
        ("creative", "✨ Creative"),
        ("van_gogh", "🖌️ Van Gogh"),
        ("illustration", "✏️ Illustration"),
        ("far_side", "🐄 Far Side"),
    ]

    STYLE_ROWS = [
        ("none",),
        ("randomize", "creative"),
        ("van_gogh", "illustration"),
        ("far_side",),
    ]

    STYLE_LABELS = {value: label for value, label in STYLE_OPTIONS}

    def _style_button(self, request_id, request, style_value):
        base_label = self.STYLE_LABELS.get(style_value, style_value.capitalize())
        current_style = request.get("style", "none")
        label = f"{base_label} ✅" if style_value == current_style else base_label
        return {
            "text": label,
            "callback_data": f"ai|{request_id}|style|{style_value}",
        }

    def __init__(
        self,
        token,
        allowed_ids,
        device_config,
        display_manager,
        refresh_task,
        poll_timeout=30,
    ):
        self.token = token
        self.allowed_ids = allowed_ids or set()
        self.device_config = device_config
        self.display_manager = display_manager
        self.refresh_task = refresh_task
        self.poll_timeout = poll_timeout

        self.api_url = self.API_BASE.format(token=token)
        self.file_url = self.FILE_BASE.format(token=token)

        self._stop_event = threading.Event()
        self._thread = None
        self._offset = None

        self.storage_dir = os.path.join(self.device_config.BASE_DIR, "..", "mock_display_output", "telegram")
        os.makedirs(self.storage_dir, exist_ok=True)

        self.pending_requests = {}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        logger.info("Starting Telegram bot polling thread")
        self._thread = threading.Thread(target=self._run, name="TelegramBotListener", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            logger.info("Telegram bot polling thread stopped")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                params = {
                    "timeout": self.poll_timeout,
                    "allowed_updates": ["message", "edited_message", "callback_query"],
                }
                if self._offset is not None:
                    params["offset"] = self._offset
                resp = requests.get(f"{self.api_url}/getUpdates", params=params, timeout=self.poll_timeout + 5)
                resp.raise_for_status()
                payload = resp.json()

                if not payload.get("ok"):
                    logger.error("Unexpected response from Telegram: %s", payload)
                    time.sleep(5)
                    continue

                for update in payload.get("result", []):
                    self._offset = update["update_id"] + 1
                    self._handle_update(update)

            except requests.RequestException as exc:
                logger.exception("Telegram polling error: %s", exc)
                time.sleep(5)
            except Exception as exc:
                logger.exception("Unexpected error in Telegram bot listener: %s", exc)
                time.sleep(5)

    def _handle_update(self, update):
        message = update.get("message") or update.get("edited_message")
        callback_query = update.get("callback_query")

        if callback_query:
            self._handle_callback(callback_query)
            return

        if not message:
            return

        chat_id = message["chat"]["id"]
        user_id = message.get("from", {}).get("id")

        if self.allowed_ids and user_id not in self.allowed_ids:
            logger.warning("Ignoring message from unauthorized user_id=%s", user_id)
            self._send_message(chat_id, "Unauthorized user.")
            return

        if "text" in message:
            self._handle_text(message["text"], chat_id, message["message_id"])
        elif "photo" in message:
            self._handle_photo(message["photo"], chat_id)
        else:
            self._send_message(chat_id, "Send a photo or use /status.")

    def _handle_text(self, text, chat_id, message_id):
        text = text.strip()
        if text.lower() in {"/start", "/help"}:
            self._send_message(
                chat_id,
                "Send a photo to update the InkyPi display, or use /status to fetch the latest rendered image.",
            )
        elif text.lower() == "/status":
            self._send_status(chat_id)
        elif text.startswith("/ai"):
            prompt = text[3:].strip()
            if not prompt:
                self._send_message(chat_id, "Usage: /ai <prompt>")
                return
            self._init_ai_prompt(chat_id, prompt)
        else:
            self._init_ai_prompt(chat_id, text)

    def _handle_photo(self, photos, chat_id):
        if not photos:
            self._send_message(chat_id, "No photo received. Please resend.")
            return

        photo = photos[-1]  # highest resolution
        file_id = photo["file_id"]

        try:
            file_info = self._api_post("getFile", {"file_id": file_id})
            file_path = file_info["result"]["file_path"]
            download_url = f"{self.file_url}/{file_path}"

            download_resp = requests.get(download_url, timeout=60)
            download_resp.raise_for_status()

            image = Image.open(BytesIO(download_resp.content)).convert("RGB")
            saved_path = self._save_image(image)
            self._display_image(image)

            self._send_message(chat_id, "Image received and sent to display.")
            logger.info("Updated display from Telegram photo %s", saved_path)
        except Exception as exc:
            logger.exception("Failed to handle Telegram photo: %s", exc)
            self._send_message(chat_id, f"Failed to update display: {exc}")

    def _display_image(self, image):
        current_dt = self.refresh_task._get_current_datetime() if hasattr(self.refresh_task, "_get_current_datetime") else datetime.utcnow()
        image_hash = compute_image_hash(image)
        self.display_manager.display_image(image)

        refresh_info = RefreshInfo(
            refresh_type="Telegram",
            plugin_id="telegram_bot",
            refresh_time=current_dt.isoformat(),
            image_hash=image_hash,
        )
        self.device_config.refresh_info = refresh_info
        self.device_config.write_config()

    def _save_image(self, image):
        filename = datetime.utcnow().strftime("telegram_%Y%m%d_%H%M%S.png")
        path = os.path.join(self.storage_dir, filename)
        image.save(path)

        latest_path = os.path.join(self.storage_dir, "latest.png")
        image.save(latest_path)
        return path

    def _send_status(self, chat_id):
        latest_path = os.path.join(self.storage_dir, "latest.png")
        if not os.path.exists(latest_path):
            self._send_message(chat_id, "No Telegram image yet. Send a photo to update the display.")
            return

        with open(latest_path, "rb") as img_file:
            files = {"photo": img_file}
            data = {"chat_id": chat_id}
            self._api_post("sendPhoto", data=data, files=files)

    def _send_message(self, chat_id, text):
        return self._api_post(
            "sendMessage",
            data={"chat_id": chat_id, "text": text},
        )

    def _api_post(self, method, data=None, files=None):
        url = f"{self.api_url}/{method}"
        resp = requests.post(url, data=data, files=files, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(payload)
        return payload

    # --- AI assistant helpers ---

    def _handle_callback(self, callback_query):
        data = callback_query.get("data", "")
        if not data:
            self._answer_callback(callback_query["id"])
            return

        parts = data.split("|")
        if len(parts) < 3 or parts[0] != "ai":
            self._answer_callback(callback_query["id"])
            return

        request_id = parts[1]
        action = parts[2]
        param = parts[3] if len(parts) > 3 else None
        request = self.pending_requests.get(request_id)
        if not request:
            self._answer_callback(callback_query["id"], text="Request expired.")
            return

        self._ensure_style(request)
        chat_id = request["chat_id"]

        if request.get("locked") and action not in ("cancel",):
            self._answer_callback(callback_query["id"], text="Processing, please wait…")
            return

        if action == "cycle_model":
            self._cycle_model(request)
            self._refresh_ai_message(request)
            self._answer_callback(callback_query["id"], text="Model updated.")
        elif action == "cycle_quality":
            self._cycle_quality(request)
            self._refresh_ai_message(request)
            self._answer_callback(callback_query["id"], text="Quality updated.")
        elif action in {"toggle_randomize", "toggle_creative", "toggle_vangogh"}:
            # Backwards compatibility for older keyboards
            mapping = {
                "toggle_randomize": "randomize",
                "toggle_creative": "creative",
                "toggle_vangogh": "van_gogh",
            }
            new_style = mapping[action]
            if request.get("style") == new_style:
                self._set_style(request, "none")
            else:
                self._set_style(request, new_style)
            self._refresh_ai_message(request)
            label = self.STYLE_LABELS.get(request["style"], request["style"])  # type: ignore[index]
            self._answer_callback(callback_query["id"], text=f"Style: {label}")
        elif action == "style" and param:
            if param not in self.STYLE_LABELS:
                self._answer_callback(callback_query["id"], text="Unknown style.")
                return
            self._set_style(request, param)
            self._refresh_ai_message(request)
            label = self.STYLE_LABELS[param]
            self._answer_callback(callback_query["id"], text=f"Style: {label}")
        elif action == "cycle_palette":
            self._cycle_palette(request)
            self._refresh_ai_message(request)
            palette_label = "Spectra 6" if request["palette"] == "spectra6" else "Monochrome"
            self._answer_callback(callback_query["id"], text=f"Palette: {palette_label}.")
        elif action == "generate":
            self._answer_callback(callback_query["id"], text="Generating image…")
            request["locked"] = True
            threading.Thread(
                target=self._process_ai_generation,
                args=(request_id,),
                name=f"TelegramAI-{request_id}",
                daemon=True,
            ).start()
        elif action == "cancel":
            self._answer_callback(callback_query["id"], text="Cancelled.")
            self._cancel_ai_request(request_id, status_text="Cancelled.")
        else:
            self._answer_callback(callback_query["id"])

    def _init_ai_prompt(self, chat_id, prompt):
        prompt = prompt.strip()
        if not prompt:
            self._send_message(chat_id, "Prompt cannot be empty.")
            return

        request_id = f"{chat_id}:{int(time.time()*1000)}"
        model = self.AI_MODELS[0][0]
        quality = self.QUALITY_OPTIONS[model][0]
        request = {
            "id": request_id,
            "chat_id": chat_id,
            "prompt": prompt,
            "model": model,
            "quality": quality,
            "palette": "spectra6",
            "message_id": None,
            "locked": False,
        }
        self._set_style(request, "none")
        self.pending_requests[request_id] = request

        summary = self._format_ai_summary(request)
        markup = self._build_ai_keyboard(request_id, request)
        response = self._api_post(
            "sendMessage",
            data={
                "chat_id": chat_id,
                "text": summary,
                "reply_markup": json.dumps(markup),
            },
        )
        request["message_id"] = response["result"]["message_id"]

    def _cycle_model(self, request):
        model_values = [m[0] for m in self.AI_MODELS]
        idx = model_values.index(request["model"])
        idx = (idx + 1) % len(model_values)
        request["model"] = model_values[idx]

        allowed_quality = self.QUALITY_OPTIONS[request["model"]]
        if request["quality"] not in allowed_quality:
            request["quality"] = allowed_quality[0]

    def _cycle_quality(self, request):
        allowed_quality = self.QUALITY_OPTIONS[request["model"]]
        idx = allowed_quality.index(request["quality"])
        idx = (idx + 1) % len(allowed_quality)
        request["quality"] = allowed_quality[idx]

    def _cycle_palette(self, request):
        request["palette"] = "bw" if request["palette"] == "spectra6" else "spectra6"

    def _set_style(self, request, style):
        if style not in self.STYLE_LABELS:
            style = "none"
        request["style"] = style
        request["randomize"] = style == "randomize"
        request["creative"] = style == "creative"
        request["van_gogh"] = style == "van_gogh"

    def _ensure_style(self, request):
        if "style" in request:
            return
        if request.get("van_gogh"):
            seed = "van_gogh"
        elif request.get("randomize"):
            seed = "randomize"
        elif request.get("creative"):
            seed = "creative"
        else:
            seed = "none"
        self._set_style(request, seed)

    def _format_ai_summary(self, request, status=None):
        model_label = dict(self.AI_MODELS)[request["model"]]
        quality_label = request["quality"].capitalize()
        style_label = self.STYLE_LABELS.get(request.get("style", "none"), "None")
        palette_label = "Spectra 6" if request["palette"] == "spectra6" else "Monochrome"
        lines = [
            "🎨 AI Image Prompt",
            "",
            f"Prompt: {request['prompt']}",
            f"Model: {model_label}",
            f"Quality: {quality_label}",
            f"Style: {style_label}",
            f"Palette: {palette_label}",
        ]
        if status:
            lines.extend(["", status])
        return "\n".join(lines)

    def _build_ai_keyboard(self, request_id, request):
        self._ensure_style(request)
        model_label = dict(self.AI_MODELS)[request["model"]]
        quality_text = request["quality"].capitalize()

        separator = {"text": "────────────", "callback_data": "ai|noop|noop"}

        keyboard = [
            [
                {
                    "text": f"⚙️ Model: {model_label}",
                    "callback_data": f"ai|{request_id}|cycle_model",
                },
                {
                    "text": f"📐 Quality: {quality_text}",
                    "callback_data": f"ai|{request_id}|cycle_quality",
                },
            ],
            [separator],
        ]

        for row in self.STYLE_ROWS:
            keyboard.append([self._style_button(request_id, request, style_value) for style_value in row])

        keyboard.append([separator])

        palette_label = "Spectra 6" if request["palette"] == "spectra6" else "Monochrome"
        keyboard.append(
            [
                {
                    "text": f"🎨 Palette: {palette_label}",
                    "callback_data": f"ai|{request_id}|cycle_palette",
                }
            ]
        )

        keyboard.append(
            [
                {
                    "text": "🚀 GENERATE IMAGE 🚀",
                    "callback_data": f"ai|{request_id}|generate",
                },
                {
                    "text": "✖️ Cancel",
                    "callback_data": f"ai|{request_id}|cancel",
                },
            ]
        )

        return {"inline_keyboard": keyboard}

    def _refresh_ai_message(self, request, status=None):
        summary = self._format_ai_summary(request, status=status)
        if request.get("locked"):
            markup = {"inline_keyboard": []}
        else:
            markup = self._build_ai_keyboard(request["id"], request)
        data = {
            "chat_id": request["chat_id"],
            "message_id": request["message_id"],
            "text": summary,
            "reply_markup": json.dumps(markup),
        }
        self._api_post("editMessageText", data=data)

    def _process_ai_generation(self, request_id):
        request = self.pending_requests.get(request_id)
        if not request:
            return

        try:
            self._refresh_ai_message(request, status="Generating image…")
        except Exception:
            logger.exception("Failed to update Telegram message during generation.")

        try:
            image = self._generate_ai_image(request)
        except Exception as exc:
            logger.exception("AI generation failed: %s", exc)
            self._send_message(request["chat_id"], f"Failed to generate image: {exc}")
            self._cancel_ai_request(request_id, status_text="Failed.")
            return

        caption = f"✅ Image generated with {dict(self.AI_MODELS)[request['model']]} ({request['quality']})"
        self._send_photo(request["chat_id"], image, caption=caption)
        self._cancel_ai_request(request_id, status_text="Completed.")

    def _cancel_ai_request(self, request_id, status_text="Cancelled."):
        request = self.pending_requests.pop(request_id, None)
        if not request:
            return

        try:
            data = {
                "chat_id": request["chat_id"],
                "message_id": request["message_id"],
                "text": self._format_ai_summary(request, status=status_text),
                "reply_markup": json.dumps({"inline_keyboard": []}),
            }
            self._api_post("editMessageText", data=data)
        except Exception:
            logger.exception("Failed to update Telegram message on completion/cancel.")

    def _generate_ai_image(self, request):
        self._ensure_style(request)
        plugin_config = self.device_config.get_plugin("ai_image")
        if not plugin_config:
            raise RuntimeError("AI Image plugin is not installed.")

        plugin = get_plugin_instance(plugin_config)
        style = request.get("style", "none")
        settings = {
            "textPrompt": request["prompt"],
            "imageModel": request["model"],
            "quality": request["quality"],
            "palette": request["palette"],
        }

        if style == "randomize":
            settings["randomizePrompt"] = "true"
        elif style == "creative":
            settings["creativeEnhance"] = "true"
        elif style == "van_gogh":
            settings["vanGoghStyle"] = "true"
            settings["styleHint"] = "van_gogh"
        elif style == "illustration":
            settings["styleHint"] = "illustration"
        elif style == "far_side":
            settings["styleHint"] = "far_side"

        image = plugin.generate_image(settings, self.device_config)
        self.display_manager.display_image(image, image_settings=plugin_config.get("image_settings", []))
        current_dt = (
            self.refresh_task._get_current_datetime()
            if hasattr(self.refresh_task, "_get_current_datetime")
            else datetime.utcnow()
        )
        image_hash = compute_image_hash(image)
        refresh_info = RefreshInfo(
            refresh_type="Telegram AI",
            plugin_id="ai_image",
            refresh_time=current_dt.isoformat(),
            image_hash=image_hash,
        )
        self.device_config.refresh_info = refresh_info
        self.device_config.write_config()

        self._save_image(image)
        return image

    def _send_photo(self, chat_id, image, caption=None):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        files = {"photo": ("image.png", buffer, "image/png")}
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        self._api_post("sendPhoto", data=data, files=files)

    def _answer_callback(self, callback_id, text=None, alert=False):
        data = {"callback_query_id": callback_id}
        if text:
            data["text"] = text
        if alert:
            data["show_alert"] = True
        # Ignore no-op callbacks used for separators
        if data.get("callback_query_id") and data.get("callback_query_id") == "noop":
            return
        self._api_post("answerCallbackQuery", data=data)
