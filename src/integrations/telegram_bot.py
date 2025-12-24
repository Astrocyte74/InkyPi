import json
import os
import threading
import time
import logging
import re
import random
from datetime import datetime, time as dt_time, timedelta
from io import BytesIO
import shutil

import requests
from requests.exceptions import HTTPError
from PIL import Image

from model import RefreshInfo
from refresh_task import PlaylistRefresh
from utils.image_utils import compute_image_hash
from plugins.plugin_registry import get_plugin_instance
from integrations.telegram_text_flow import TelegramTextFlow

logger = logging.getLogger(__name__)


class TelegramBotListener:
    """Background poller that listens for Telegram messages and triggers display updates."""

    API_BASE = "https://api.telegram.org/bot{token}"
    FILE_BASE = "https://api.telegram.org/file/bot{token}"

    AI_MODELS = [
        ("gemini-2.5-flash-image", "Gemini 2.5 Flash Image"),
        ("gemini-3-pro-image-preview", "Gemini 3 Pro Image (Preview)"),
        ("dall-e-3", "DALL·E 3"),
        ("gpt-image-1", "GPT Image 1"),
        ("dall-e-2", "DALL·E 2"),
    ]

    QUALITY_OPTIONS = {
        "dall-e-3": ["standard", "hd"],
        "gpt-image-1": ["medium", "high", "low"],
        "dall-e-2": ["standard"],
        # Gemini quality maps to image_size (1K / 2K / 4K) in google-genai.
        "gemini-2.5-flash-image": ["2k", "1k", "4k"],
        "gemini-3-pro-image-preview": ["2k", "1k", "4k"],
    }

    STYLE_OPTIONS = [
        ("none", "🚫 None"),
        ("randomize", "🎲 Randomize"),
        ("creative", "✨ Creative"),
        ("van_gogh", "🖌️ Van Gogh"),
        ("illustration", "✏️ Illustration"),
        ("drawing", "📝 Drawing"),
        ("far_side", "🐄 Far Side"),
    ]

    STYLE_ROWS = [
        ("none",),
        ("randomize", "creative"),
        ("van_gogh", "illustration"),
        ("drawing", "far_side"),
    ]

    STYLE_LABELS = {value: label for value, label in STYLE_OPTIONS}
    PALETTE_LABELS = {
        "spectra6": "Colour",
        "bw": "Black & White",
    }

    @staticmethod
    def _resolve_default_model(env_value, model_values, fallback):
        """Resolve TELEGRAM_AI_DEFAULT_MODEL which may contain a comma/space-separated list."""
        if not env_value:
            return fallback

        candidates = [token for token in re.split(r"[,\\s]+", env_value.strip()) if token]
        if not candidates:
            return fallback

        selected = candidates[0]
        return selected if selected in model_values else fallback

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
        self.text_flow = TelegramTextFlow(self.device_config, self.display_manager, self.refresh_task, self.storage_dir)
        self.pending_save = {}
        self.pending_txt_bg = {}
        self.pending_help_prompt = {}
        self.load_filters = {}
        self.load_wx = {}
        self.wx_menu_ids = {}
        self.ss_menu_ids = {}
        self.pending_cat_reroll = set()
        self.pending_card_updates = set()
        self.pending_card_bg_updates = set()
        # Slideshow state
        self.slideshow_thread = None
        self.slideshow_stop = threading.Event()
        self.slideshow_state = {
            "active": False,
            "filter": "all",   # all|bg|composite
            "interval": 60,     # seconds
            "shuffle": False,
            "weather": "off",  # off|badge|overlay
            "was_refresh_running": False,
        }

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

        # Handle pending /save name entry first (takes precedence over flow prompts)
        if self.pending_save.get(chat_id):
            save_ctx = self.pending_save.pop(chat_id)
            name = text
            src = save_ctx.get("source", "bg")
            try:
                saved_path = self._save_named_image(name, src)
                self._send_message(chat_id, f"Saved {src} image as '{os.path.basename(saved_path)}'.")
            except Exception as exc:
                logger.exception("Failed to save named image: %s", exc)
                self._send_message(chat_id, f"Save failed: {exc}")
            return

        pending_request = self.text_flow.consume_prompt(chat_id, text)
        if pending_request:
            # If the current background mode is custom, switch into AI image flow now
            if pending_request.get("background") == "custom_ai":
                # Do not edit the original message here to avoid UI shrinking in Telegram
                self._start_custom_background_flow(pending_request)
            else:
                self._refresh_text_message(pending_request, status="Background prompt updated.")
            return

        # Handle pending /load -> Use as Background message entry
        if self.pending_txt_bg.get(chat_id):
            saved_name = self.pending_txt_bg.pop(chat_id)
            message = text
            request = self.text_flow.create_request(chat_id, message)
            self.text_flow.set_background(request, "saved")
            self.text_flow.set_saved_name(request, saved_name)
            summary = self.text_flow.format_summary(request)
            markup = self.text_flow.build_keyboard(request)
            self._api_post(
                "sendMessage",
                data={
                    "chat_id": chat_id,
                    "text": summary,
                    "reply_markup": json.dumps(markup),
                },
            )
            return

        pending_help = self.pending_help_prompt.pop(chat_id, None)
        if pending_help and not text.startswith("/"):
            if pending_help == "ai":
                self._init_ai_prompt(chat_id, text)
            elif pending_help == "txt":
                self._init_text_prompt(chat_id, text)
            return

        if text.lower() in {"/start", "/help"}:
            self._send_help(chat_id)
        elif text.lower() == "/status":
            self._send_status(chat_id)
        elif text.lower().startswith("/load"):
            parts = text.split(maxsplit=2)
            name = parts[1].strip() if len(parts) >= 2 else None
            if name:
                path = os.path.join(self.text_flow.storage_dir, "saved", f"{name}.png")
                if not os.path.exists(path):
                    self._send_message(chat_id, f"Saved image '{name}' not found.")
                    return
                try:
                    with Image.open(path) as img:
                        self.display_manager.display_image(img)
                    self._send_message(chat_id, f"Loaded '{name}' to display.")
                except Exception as exc:
                    logger.exception("Failed to load saved image: %s", exc)
                    self._send_message(chat_id, f"Load failed: {exc}")
            else:
                self._send_load_menu(chat_id)
            return
        elif text.lower().startswith("/save"):
            # Usage: /save [bg|text] <name>
            parts = text.split(maxsplit=2)
            source = "bg"
            name = None
            if len(parts) == 2:
                # Could be a name or a source keyword
                if parts[1].lower() in {"bg", "background", "image"}:
                    source = "bg"
                elif parts[1].lower() in {"text", "overlay"}:
                    source = "text"
                else:
                    name = parts[1]
            elif len(parts) >= 3:
                if parts[1].lower() in {"bg", "background", "image"}:
                    source = "bg"
                elif parts[1].lower() in {"text", "overlay"}:
                    source = "text"
                name = parts[2]

            if name:
                try:
                    saved_path = self._save_named_image(name, source)
                    self._send_message(chat_id, f"Saved {source} image as '{os.path.basename(saved_path)}'.")
                except Exception as exc:
                    logger.exception("Failed to save named image: %s", exc)
                    self._send_message(chat_id, f"Save failed: {exc}")
            else:
                # Show interactive save menu instead of ForceReply
                self._send_save_menu(chat_id)
            return
        elif text.startswith("/ai"):
            prompt = text[3:].strip()
            if not prompt:
                self._send_message(chat_id, "Usage: /ai <prompt>")
                return
            self._init_ai_prompt(chat_id, prompt)
        elif text.lower().startswith("/txt"):
            message = text[4:].strip()
            if not message:
                self._send_message(chat_id, "Usage: /txt <message>")
                return
            self._init_text_prompt(chat_id, message)
        elif text.lower().startswith("/weather") or text.lower().startswith("/wx"):
            self._send_weather_menu(chat_id)
        elif text.lower().startswith("/theme"):
            parts = text.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else ""
            if not arg:
                self._send_daily_theme_dashboard(chat_id)
            else:
                self._set_daily_cat_theme(chat_id, arg)
        elif text.lower().startswith("/card"):
            match = re.match(r"^/card(\d+)(?:\s+(.*))?$", text.strip(), flags=re.IGNORECASE)
            if not match:
                self._send_message(chat_id, "Usage: /card1, /card2, /card3 … (e.g. `/card1 word_of_day`).")
                return
            slot = int(match.group(1))
            arg = (match.group(2) or "").strip()
            if not arg:
                self._send_daily_card_menu(chat_id, slot=slot)
            else:
                self._set_daily_card(chat_id, slot=slot, raw_card=arg)
        elif text.lower().startswith("/cat") or text.lower().startswith("/reroll"):
            parts = text.split(maxsplit=1)
            user_prompt = parts[1].strip() if len(parts) > 1 else None
            self._reroll_daily_cat(chat_id, user_prompt=user_prompt)
        elif text.lower().startswith("/slideshow"):
            self._handle_slideshow_command(text, chat_id)
        elif text.lower().strip() == "/stop":
            self._stop_slideshow(chat_id)
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
        # Prefer the actual currently displayed image (works even when the image came from playlists/plugins).
        current_path = getattr(self.device_config, "current_image_file", None)
        if current_path and os.path.exists(current_path):
            caption = "Current display"
            try:
                refresh_info = self.device_config.get_refresh_info()
            except Exception:
                refresh_info = None

            try:
                if getattr(refresh_info, "plugin_id", None) == "daily_cat_weather":
                    theme_id = None
                    try:
                        playlist_manager = self.device_config.get_playlist_manager()
                        instance_name = getattr(refresh_info, "plugin_instance", None)
                        plugin_instance = (
                            playlist_manager.find_plugin("daily_cat_weather", instance_name)
                            if instance_name
                            else None
                        )
                        theme_id = (plugin_instance.settings or {}).get("imageTheme") if plugin_instance else None
                    except Exception:
                        theme_id = None

                    theme_id = (theme_id or "storybook").strip().lower()
                    presets = self._daily_cat_theme_presets()
                    theme_label = (presets.get(theme_id, {}).get("label") if presets else None) or theme_id
                    caption = f"🎨 Daily Theme (theme={theme_id} — {theme_label})"
                elif getattr(refresh_info, "plugin_id", None) == "daily_theme_card":
                    card_id = None
                    try:
                        playlist_manager = self.device_config.get_playlist_manager()
                        instance_name = getattr(refresh_info, "plugin_instance", None)
                        plugin_instance = (
                            playlist_manager.find_plugin("daily_theme_card", instance_name)
                            if instance_name
                            else None
                        )
                        card_id = (plugin_instance.settings or {}).get("cardId") if plugin_instance else None
                    except Exception:
                        card_id = None

                    card_id = (card_id or "inspiration").strip().lower()
                    presets = self._daily_card_presets()
                    label = (presets.get(card_id, {}).get("label") if presets else None) or card_id
                    caption = f"🗒️ Card 1 (card={card_id} — {label})"
            except Exception:
                caption = "Current display"
            self._send_photo_path(chat_id, current_path, caption=caption)
            return

        # Fallback: last Telegram-updated background
        latest_path = os.path.join(self.storage_dir, "latest.png")
        if os.path.exists(latest_path):
            self._send_photo_path(chat_id, latest_path, caption="Last Telegram background")
            return

        self._send_message(chat_id, "No image available yet.")

    def _send_message(self, chat_id, text):
        return self._api_post(
            "sendMessage",
            data={"chat_id": chat_id, "text": text},
        )

    def _sanitize_name(self, name):
        cleaned = "".join(c if c.isalnum() or c in {"-", "_"} else "-" for c in name.strip())
        while "--" in cleaned:
            cleaned = cleaned.replace("--", "-")
        cleaned = cleaned.strip("-")
        if not cleaned:
            raise ValueError("Invalid name.")
        return cleaned.lower()

    def _save_named_image(self, name, source="bg"):
        safe = self._sanitize_name(name)
        saved_dir = os.path.join(self.storage_dir, "saved")
        os.makedirs(saved_dir, exist_ok=True)
        if source == "text":
            latest = os.path.join(self.storage_dir, "latest_text.png")
        elif source == "txtbg":
            latest = os.path.join(self.storage_dir, "last_text_background.png")
        else:
            latest = os.path.join(self.storage_dir, "latest.png")
        if not os.path.exists(latest):
            raise FileNotFoundError("No recent image to save.")
        target = os.path.join(saved_dir, f"{safe}.png")
        final_target = target
        suffix = 1
        while os.path.exists(final_target):
            final_target = os.path.join(saved_dir, f"{safe}-{suffix}.png")
            suffix += 1
        shutil.copyfile(latest, final_target)
        logger.info("Saved %s image to %s", source, final_target)
        return final_target

    def _api_post(self, method, data=None, files=None):
        url = f"{self.api_url}/{method}"
        resp = requests.post(url, data=data, files=files, timeout=60)
        try:
            resp.raise_for_status()
        except HTTPError as exc:
            payload = {}
            try:
                payload = resp.json()
            except ValueError:
                pass
            description = (payload.get("description") or "").lower()
            if "message is not modified" in description:
                logger.debug("Telegram API ignored edit: %s", payload.get("description"))
                return payload
            raise
        payload = resp.json()
        if not payload.get("ok"):
            description = (payload.get("description") or "").lower()
            if "message is not modified" in description:
                logger.debug("Telegram API ignored edit: %s", payload.get("description"))
                return payload
            raise RuntimeError(payload)
        return payload

    # --- AI assistant helpers ---
    @staticmethod
    def _trim_text(text, limit=240):
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _handle_callback(self, callback_query):
        data = callback_query.get("data", "")
        if not data:
            self._answer_callback(callback_query["id"])
            return

        parts = data.split("|")
        if len(parts) < 2:
            self._answer_callback(callback_query["id"])
            return

        flow_type = parts[0]
        if flow_type == "help":
            action = parts[1] if len(parts) > 1 else None
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")

            if not chat_id:
                self._answer_callback(callback_query["id"])
                return

            if action == "status":
                self._send_status(chat_id)
                self._answer_callback(callback_query["id"])
            elif action == "cat":
                self._reroll_daily_cat(chat_id)
                self._answer_callback(callback_query["id"], text="Refreshing…")
            elif action == "theme":
                self._send_daily_theme_dashboard(chat_id, message_id=message_id)
                self._answer_callback(callback_query["id"])
            elif action == "ai":
                self.pending_help_prompt[chat_id] = "ai"
                self._api_post(
                    "sendMessage",
                    data={
                        "chat_id": chat_id,
                        "text": "Send an AI image prompt (or run `/ai <prompt>`).",
                        "reply_markup": json.dumps(
                            {"force_reply": True, "input_field_placeholder": "AI prompt"}
                        ),
                    },
                )
                self._answer_callback(callback_query["id"], text="Send a prompt…")
            elif action == "txt":
                self.pending_help_prompt[chat_id] = "txt"
                self._api_post(
                    "sendMessage",
                    data={
                        "chat_id": chat_id,
                        "text": "Send a short note (or run `/txt <message>`).",
                        "reply_markup": json.dumps(
                            {"force_reply": True, "input_field_placeholder": "Note text"}
                        ),
                    },
                )
                self._answer_callback(callback_query["id"], text="Send a note…")
            elif action == "save":
                self._send_save_menu(chat_id)
                self._answer_callback(callback_query["id"])
            elif action == "load":
                self._send_load_menu(chat_id)
                self._answer_callback(callback_query["id"])
            elif action == "weather":
                self._send_weather_menu(chat_id)
                self._answer_callback(callback_query["id"])
            elif action == "slideshow":
                self._send_slideshow_menu(chat_id)
                self._answer_callback(callback_query["id"])
            elif action == "close" and message_id:
                try:
                    self._api_post(
                        "editMessageReplyMarkup",
                        data={
                            "chat_id": chat_id,
                            "message_id": message_id,
                            "reply_markup": json.dumps({"inline_keyboard": []}),
                        },
                    )
                except Exception:
                    logger.exception("Failed to close help menu.")
                self._answer_callback(callback_query["id"])
            else:
                self._answer_callback(callback_query["id"])
            return

        if flow_type == "cat_theme":
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")
            if not chat_id:
                self._answer_callback(callback_query["id"])
                return
            action = parts[1] if len(parts) > 1 else None
            if action == "set" and len(parts) >= 3:
                self._set_daily_cat_theme(chat_id, parts[2], menu_message_id=message_id, include_back=True)
                self._answer_callback(callback_query["id"], text="Setting theme…")
            else:
                self._answer_callback(callback_query["id"])
            return

        if flow_type == "daily":
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")
            if not chat_id:
                self._answer_callback(callback_query["id"])
                return

            action = parts[1] if len(parts) > 1 else None
            target = parts[2] if len(parts) > 2 else None
            if action == "open" and target == "dashboard":
                self._send_daily_theme_dashboard(chat_id, message_id=message_id)
                self._answer_callback(callback_query["id"])
            elif action == "open" and target == "illustration":
                self._send_daily_cat_theme_menu(chat_id, message_id=message_id, include_back=True)
                self._answer_callback(callback_query["id"])
            elif action == "open" and target == "card":
                try:
                    slot = int(parts[3]) if len(parts) > 3 else 1
                except (TypeError, ValueError):
                    slot = 1
                group_id = ""
                if len(parts) > 5 and parts[4] == "group":
                    group_id = parts[5]
                elif len(parts) > 4:
                    group_id = parts[4]
                self._send_daily_card_menu(
                    chat_id,
                    slot=slot,
                    message_id=message_id,
                    include_back=True,
                    group_id=group_id,
                )
                self._answer_callback(callback_query["id"])
            elif action == "open" and target == "cardbg":
                try:
                    slot = int(parts[3]) if len(parts) > 3 else 1
                except (TypeError, ValueError):
                    slot = 1
                self._send_daily_card_background_menu(chat_id, slot=slot, message_id=message_id, include_back=True)
                self._answer_callback(callback_query["id"])
            else:
                self._answer_callback(callback_query["id"])
            return

        if flow_type == "cardbg":
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")
            if not chat_id:
                self._answer_callback(callback_query["id"])
                return
            try:
                slot = int(parts[1]) if len(parts) > 1 else 1
            except (TypeError, ValueError):
                slot = 1
            action = parts[2] if len(parts) > 2 else None
            if action == "set" and len(parts) >= 4:
                mode = parts[3]
                self._set_daily_card_background(
                    chat_id,
                    slot=slot,
                    mode=mode,
                    menu_message_id=message_id,
                    include_back=True,
                )
                self._answer_callback(callback_query["id"], text="Updating…")
            else:
                self._answer_callback(callback_query["id"])
            return

        if flow_type == "card":
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")
            if not chat_id:
                self._answer_callback(callback_query["id"])
                return
            try:
                slot = int(parts[1]) if len(parts) > 1 else 1
            except (TypeError, ValueError):
                slot = 1
            action = parts[2] if len(parts) > 2 else None
            if action == "set" and len(parts) >= 4:
                self._set_daily_card(chat_id, slot=slot, raw_card=parts[3], menu_message_id=message_id, include_back=True)
                self._answer_callback(callback_query["id"], text="Setting card…")
            else:
                self._answer_callback(callback_query["id"])
            return

        # Backwards compatibility for older menus
        if flow_type == "card1":
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")
            if not chat_id:
                self._answer_callback(callback_query["id"])
                return
            action = parts[1] if len(parts) > 1 else None
            if action == "set" and len(parts) >= 3:
                self._set_daily_card(chat_id, slot=1, raw_card=parts[2], menu_message_id=message_id, include_back=True)
                self._answer_callback(callback_query["id"], text="Setting card…")
            else:
                self._answer_callback(callback_query["id"])
            return

        if flow_type == "ai":
            if len(parts) < 3:
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
                palette_label = self._get_palette_label(request["palette"])
                self._answer_callback(callback_query["id"], text=f"Palette: {palette_label}.")
            elif action == "wbadge" and param:
                # Single-select: Badge ON implies Overlay OFF
                request["wbadge"] = (param == "on")
                if request["wbadge"]:
                    request["woverlay"] = False
                self._refresh_ai_message(request)
                self._answer_callback(callback_query["id"], text=f"Weather: {'Badge' if request.get('wbadge') else 'Off'}")
            elif action == "wover" and param:
                # Single-select: Overlay ON implies Badge OFF
                request["woverlay"] = (param == "on")
                if request["woverlay"]:
                    request["wbadge"] = False
                self._refresh_ai_message(request)
                self._answer_callback(callback_query["id"], text=f"Weather: {'Overlay' if request.get('woverlay') else 'Off'}")
            elif action == "woff":
                request["wbadge"] = False
                request["woverlay"] = False
                self._refresh_ai_message(request)
                self._answer_callback(callback_query["id"], text="Weather: Off")
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
            elif action == "open_wx":
                chat_id = callback_query["message"]["chat"]["id"]
                self._send_weather_menu(chat_id)
                self._answer_callback(callback_query["id"], text="Weather options opened.")
            else:
                self._answer_callback(callback_query["id"])
        elif flow_type == "txt":
            request_id = parts[1]
            action = parts[2] if len(parts) > 2 else None
            param = parts[3] if len(parts) > 3 else None
            request = self.text_flow.get_request(request_id)
            if not request:
                self._answer_callback(callback_query["id"], text="Text request expired.")
                return

            if request.get("locked") and action not in {"cancel"}:
                self._answer_callback(callback_query["id"], text="Processing, please wait…")
                return

            if action == "cycle_style":
                # Backward compatibility: still support cycle_style if old keyboards exist
                self.text_flow.cycle_style(request)
                self._refresh_text_message(request)
                label = dict(self.text_flow.STYLE_OPTIONS).get(request["style"], request["style"])
                self._answer_callback(callback_query["id"], text=f"Style: {label}")
            elif action == "style" and param:
                self.text_flow.set_style(request, param)
                self._refresh_text_message(request)
                label = dict(self.text_flow.STYLE_OPTIONS).get(request["style"], request["style"])
                self._answer_callback(callback_query["id"], text=f"Style: {label}")
            elif action == "toggle_rewrite":
                self.text_flow.toggle_rewrite(request)
                status = "On" if request["rewrite"] else "Off"
                self._refresh_text_message(request)
                self._answer_callback(callback_query["id"], text=f"Rewrite: {status}")
            elif action == "rewrite" and param:
                self.text_flow.set_rewrite(request, param == "on")
                status = "On" if request["rewrite"] else "Off"
                self._refresh_text_message(request)
                self._answer_callback(callback_query["id"], text=f"Rewrite: {status}")
            elif action == "wbadge" and param:
                # Single-select: Badge ON implies Overlay OFF
                self.text_flow.set_wbadge(request, param == "on")
                if request.get("wbadge"):
                    self.text_flow.set_woverlay(request, False)
                self._refresh_text_message(request)
                self._answer_callback(callback_query["id"], text=f"Weather: {'Badge' if request.get('wbadge') else 'Off'}")
            elif action == "wover" and param:
                # Single-select: Overlay ON implies Badge OFF
                self.text_flow.set_woverlay(request, param == "on")
                if request.get("woverlay"):
                    self.text_flow.set_wbadge(request, False)
                self._refresh_text_message(request)
                self._answer_callback(callback_query["id"], text=f"Weather: {'Overlay' if request.get('woverlay') else 'Off'}")
            elif action == "woff":
                self.text_flow.set_wbadge(request, False)
                self.text_flow.set_woverlay(request, False)
                self._refresh_text_message(request)
                self._answer_callback(callback_query["id"], text="Weather: Off")
            elif action == "cycle_background":
                # Backward compatibility
                self.text_flow.cycle_background(request)
                label = dict(self.text_flow.BACKGROUND_OPTIONS).get(request["background"], request["background"])
                self._refresh_text_message(request)
                self._answer_callback(callback_query["id"], text=f"Background: {label}")
            elif action == "background" and param:
                self.text_flow.set_background(request, param)
                label = dict(self.text_flow.BACKGROUND_OPTIONS).get(request["background"], request["background"])
                if request.get("background") == "custom_ai":
                    # Immediately request a custom prompt to reduce steps; avoid editing original message
                    self.text_flow.await_custom_prompt(request)
                    try:
                        self._api_post(
                            "sendMessage",
                            data={
                                "chat_id": request["chat_id"],
                                "text": "Enter a background prompt for the image (or /skip).",
                                "reply_markup": json.dumps({"force_reply": True, "input_field_placeholder": "Type background prompt"}),
                            },
                        )
                    except Exception:
                        logger.exception("Failed to send ForceReply prompt after background select.")
                elif request.get("background") == "saved":
                    # Show picker via inline keyboard
                    self._refresh_text_message(request)
                else:
                    self._refresh_text_message(request)
                self._answer_callback(callback_query["id"], text=f"Background: {label}")
            elif action == "saved_pick" and param:
                self.text_flow.set_saved_name(request, param)
                self._refresh_text_message(request)
                self._answer_callback(callback_query["id"], text=f"Saved: {param}")
            elif action == "saved_page" and param:
                try:
                    page = int(param)
                except ValueError:
                    page = 0
                request["saved_page"] = max(0, page)
                self._refresh_text_message(request)
                self._answer_callback(callback_query["id"], text=f"Page {request['saved_page']+1}")
            elif action == "saved_enter":
                self.text_flow.await_saved(request)
                try:
                    self._api_post(
                        "sendMessage",
                        data={
                            "chat_id": request["chat_id"],
                            "text": "Enter saved image name (from /save)",
                            "reply_markup": json.dumps({"force_reply": True, "input_field_placeholder": "saved-name"}),
                        },
                    )
                except Exception:
                    logger.exception("Failed to send ForceReply for saved image name.")
                # Do not edit original message here
                self._answer_callback(callback_query["id"], text="Awaiting name…")
            elif action == "saved_delete":
                name = (request.get("saved_name") or "").strip()
                if not name:
                    self._answer_callback(callback_query["id"], text="Pick an image first.")
                    return
                try:
                    self.text_flow.delete_saved(name)
                    request["saved_name"] = None
                    self._refresh_text_message(request, status="Deleted.")
                    self._answer_callback(callback_query["id"], text="Deleted.")
                except Exception as exc:
                    logger.exception("Delete saved failed: %s", exc)
                    self._answer_callback(callback_query["id"], text="Delete failed.")
            elif action == "saved_rename":
                name = (request.get("saved_name") or "").strip()
                if not name:
                    self._answer_callback(callback_query["id"], text="Pick an image first.")
                    return
                request["awaiting_saved_rename"] = True
                request["saved_rename_from"] = name
                try:
                    # Show placeholder without internal prefix
                    base = self.text_flow._strip_prefix(name)  # pylint: disable=protected-access
                    self._api_post(
                        "sendMessage",
                        data={
                            "chat_id": request["chat_id"],
                            "text": f"Enter new name for '{base}':",
                            "reply_markup": json.dumps({"force_reply": True, "input_field_placeholder": base}),
                        },
                    )
                except Exception:
                    logger.exception("Failed to send ForceReply for rename.")
                self._answer_callback(callback_query["id"], text="Awaiting new name…")
            elif action == "saved_preview":
                # Send the selected saved image as a quick preview
                name = (request.get("saved_name") or "").strip()
                if not name:
                    self._answer_callback(callback_query["id"], text="Pick an image first.")
                    return
                path = os.path.join(self.text_flow.storage_dir, "saved", f"{name}.png")
                try:
                    with Image.open(path) as img:
                        self._send_photo(request["chat_id"], img, caption=f"Preview: {name}")
                    self._answer_callback(callback_query["id"], text="Preview sent.")
                except Exception as exc:
                    logger.exception("Failed to send saved image preview: %s", exc)
                    self._answer_callback(callback_query["id"], text="Preview failed.")
            elif action == "saved_clear":
                request["saved_name"] = None
                self._refresh_text_message(request, status="Cleared.")
                self._answer_callback(callback_query["id"], text="Cleared.")
            elif action == "bg_color" and param:
                self.text_flow.set_bg_color(request, param)
                self._refresh_text_message(request)
                self._answer_callback(callback_query["id"], text="Colour selected.")
            elif action == "noop":
                # No operation, just dismiss the callback
                self._answer_callback(callback_query["id"])            
            elif action == "set_prompt":
                self.text_flow.await_custom_prompt(request)
                try:
                    self._api_post(
                        "sendMessage",
                        data={
                            "chat_id": request["chat_id"],
                            "text": "Reply with a background prompt (or /skip to reuse the note).",
                            "reply_markup": json.dumps({"force_reply": True, "input_field_placeholder": "Type background prompt"}),
                        },
                    )
                except Exception:
                    logger.exception("Failed to send ForceReply prompt request.")
                # Do not edit original message; avoid visual shrink
                self._answer_callback(callback_query["id"], text="Send background prompt…")
            elif action == "confirm":
                if not request.get("bg_selected"):
                    self._refresh_text_message(request, status="Choose a background first.")
                    self._answer_callback(callback_query["id"], text="Select a background option.")
                    return
                if request.get("background") == "custom_ai" and request.get("awaiting_prompt"):
                    self._refresh_text_message(request, status="Enter a background prompt before continuing.")
                    self._answer_callback(callback_query["id"], text="Send a background prompt first.")
                    return
                request["locked"] = True
                self._refresh_text_message(request, status="Rendering…")
                self._answer_callback(callback_query["id"], text="Rendering…")
                threading.Thread(
                    target=self._process_text_request,
                    args=(request_id,),
                    name=f"TelegramText-{request_id}",
                    daemon=True,
                ).start()
            elif action == "cancel":
                self._refresh_text_message(request, status="Cancelled.")
                self.text_flow.cancel_request(request_id)
                self._answer_callback(callback_query["id"], text="Cancelled.")
            else:
                self._answer_callback(callback_query["id"])
        elif flow_type == "wx":
            action = parts[1] if len(parts) > 1 else None
            arg = parts[2] if len(parts) > 2 else None
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")
            opts = self._get_weather_options()
            if action == "open":
                self._send_weather_menu(chat_id)
                self._answer_callback(callback_query["id"], text="Weather options opened.")
            elif action == "badge" and arg in {"on", "off"}:
                enabled = arg == "on"
                opts.setdefault("weather", {}).setdefault("badge", {})["enabled"] = enabled
                self._set_weather_options(opts)
                self._refresh_weather_menu(chat_id, message_id)
                self._answer_callback(callback_query["id"], text=f"Badge: {'On' if enabled else 'Off'}")
            elif action == "pos" and arg == "cycle":
                pos = opts.get("weather", {}).get("badge", {}).get("position", "tr").lower()
                order = ["tr", "tl", "bl", "br"]
                try:
                    idx = order.index(pos)
                except ValueError:
                    idx = 0
                new_pos = order[(idx + 1) % len(order)]
                opts.setdefault("weather", {}).setdefault("badge", {})["position"] = new_pos
                self._set_weather_options(opts)
                self._refresh_weather_menu(chat_id, message_id)
                self._answer_callback(callback_query["id"], text=f"Badge pos: {new_pos.upper()}")
            elif action == "overlay" and arg in {"on", "off"}:
                enabled = arg == "on"
                opts.setdefault("weather", {}).setdefault("overlay", {})["enabled"] = enabled
                self._set_weather_options(opts)
                self._refresh_weather_menu(chat_id, message_id)
                self._answer_callback(callback_query["id"], text=f"Overlay: {'On' if enabled else 'Off'}")
            elif action == "close":
                try:
                    self._api_post("editMessageReplyMarkup", data={
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "reply_markup": json.dumps({"inline_keyboard": []}),
                    })
                except Exception:
                    logger.exception("Failed to close weather menu")
                self._answer_callback(callback_query["id"]) 
            else:
                self._answer_callback(callback_query["id"]) 
        elif flow_type == "ss":
            action = parts[1] if len(parts) > 1 else None
            arg = parts[2] if len(parts) > 2 else None
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")
            try:
                if action == "open":
                    self._send_slideshow_menu(chat_id)
                    self._answer_callback(callback_query["id"]) 
                elif action == "cycle_filter":
                    order = ["all", "bg", "composite"]
                    cur = self.slideshow_state.get("filter", "all")
                    try:
                        idx = order.index(cur)
                    except ValueError:
                        idx = 0
                    self.slideshow_state["filter"] = order[(idx + 1) % len(order)]
                    self._refresh_slideshow_menu(chat_id, message_id)
                    self._answer_callback(callback_query["id"], text=f"Filter: {self.slideshow_state['filter']}")
                elif action == "interval" and arg in {"inc", "dec"}:
                    delta = 10 if arg == "inc" else -10
                    cur = int(self.slideshow_state.get("interval", 60))
                    self.slideshow_state["interval"] = max(5, cur + delta)
                    self._refresh_slideshow_menu(chat_id, message_id)
                    self._answer_callback(callback_query["id"], text=f"Interval: {self.slideshow_state['interval']}s")
                elif action == "toggle_shuffle":
                    self.slideshow_state["shuffle"] = not bool(self.slideshow_state.get("shuffle"))
                    self._refresh_slideshow_menu(chat_id, message_id)
                    self._answer_callback(callback_query["id"], text=f"Shuffle: {'On' if self.slideshow_state['shuffle'] else 'Off'}")
                elif action == "cycle_weather":
                    order = ["off", "badge", "overlay"]
                    cur = self.slideshow_state.get("weather", "off")
                    try:
                        idx = order.index(cur)
                    except ValueError:
                        idx = 0
                    self.slideshow_state["weather"] = order[(idx + 1) % len(order)]
                    self._refresh_slideshow_menu(chat_id, message_id)
                    self._answer_callback(callback_query["id"], text=f"Weather: {self.slideshow_state['weather']}")
                elif action == "start":
                    # Use current staged values
                    st = self.slideshow_state
                    self._start_slideshow(chat_id, filt=st.get('filter','all'), interval=int(st.get('interval',60)), shuffle=bool(st.get('shuffle')), weather=st.get('weather','off'))
                    self._refresh_slideshow_menu(chat_id, message_id)
                    self._answer_callback(callback_query["id"], text="Slideshow starting")
                elif action == "stop":
                    self._stop_slideshow(chat_id)
                    self._refresh_slideshow_menu(chat_id, message_id)
                    self._answer_callback(callback_query["id"], text="Slideshow stopped")
                elif action == "close":
                    try:
                        self._api_post("editMessageReplyMarkup", data={
                            "chat_id": chat_id,
                            "message_id": message_id,
                            "reply_markup": json.dumps({"inline_keyboard": []}),
                        })
                    except Exception:
                        logger.exception("Failed to close slideshow menu")
                    self._answer_callback(callback_query["id"]) 
                else:
                    self._answer_callback(callback_query["id"]) 
            except Exception:
                logger.exception("Slideshow menu handling failed")
                self._answer_callback(callback_query["id"]) 
        elif flow_type == "save":
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")
            action = parts[1] if len(parts) > 1 else None
            arg = parts[2] if len(parts) > 2 else None
            if action == "choose" and arg in {"bg", "text", "txtbg", "bgauto"}:
                # Resolve auto background to last /txt background if available
                eff_source = arg
                if arg == "bgauto":
                    last_bg = os.path.join(self.text_flow.storage_dir, "last_text_background.png")
                    eff_source = "txtbg" if os.path.exists(last_bg) else "bg"
                suggestion = self._suggest_save_name(eff_source)
                pretty = (
                    "Background (Auto)" if arg == "bgauto" else (
                        "Background" if eff_source == "bg" else (
                            "Last Background (/txt)" if eff_source == "txtbg" else "Last Background and Text"
                        )
                    )
                )
                text = f"Save {pretty}\n\nSuggested name: {suggestion}"
                markup = {
                    "inline_keyboard": [
                        [
                            {"text": f"Quick Save as '{suggestion}'", "callback_data": f"save|quick|{eff_source}|{suggestion}"},
                        ],
                        [
                            {"text": "📝 Enter Name…", "callback_data": f"save|prompt|{eff_source}"},
                            {"text": "⬅️ Back", "callback_data": "save|back"},
                            {"text": "✖️ Cancel", "callback_data": "save|cancel"},
                        ],
                    ]
                }
                self._refresh_save_message(chat_id, message_id, text, markup)
                self._answer_callback(callback_query["id"]) 
            elif action == "quick" and arg in {"bg", "text", "txtbg"}:
                # parts[3] suggestion
                name = parts[3] if len(parts) > 3 else self._suggest_save_name(arg)
                try:
                    saved_path = self._save_named_image(name, arg)
                    text = f"Saved {arg} image as '{os.path.basename(saved_path)}'."
                    self._refresh_save_message(chat_id, message_id, text, {"inline_keyboard": []})
                    self._answer_callback(callback_query["id"], text="Saved.")
                except Exception as exc:
                    logger.exception("Quick save failed: %s", exc)
                    self._answer_callback(callback_query["id"], text="Save failed.")
            elif action == "prompt" and arg in {"bg", "text", "txtbg"}:
                # Ask name via ForceReply
                try:
                    self.pending_save[chat_id] = {"source": arg}
                    self._api_post(
                        "sendMessage",
                        data={
                            "chat_id": chat_id,
                            "text": f"Enter a name to save the {arg} image:",
                            "reply_markup": json.dumps({"force_reply": True, "input_field_placeholder": self._suggest_save_name(arg)}),
                        },
                    )
                    self._answer_callback(callback_query["id"], text="Awaiting name…")
                except Exception:
                    logger.exception("Failed to send ForceReply for /save name.")
            elif action == "back":
                self._refresh_save_message(chat_id, message_id, "Save Image\n\nPick which image to save and optionally choose a name.", {
                    "inline_keyboard": [
                        [
                            {"text": "Save Background (Auto)", "callback_data": "save|choose|bgauto"},
                            {"text": "Save Last Background and Text", "callback_data": "save|choose|text"},
                        ],
                        [
                            {"text": "✖️ Cancel", "callback_data": "save|cancel"},
                        ],
                    ]
                })
                self._answer_callback(callback_query["id"]) 
            elif action == "cancel":
                # Close out the save UI
                self._refresh_save_message(chat_id, message_id, "Save cancelled.", {"inline_keyboard": []})
                self._answer_callback(callback_query["id"], text="Cancelled.")
            else:
                self._answer_callback(callback_query["id"]) 
        elif flow_type == "load":
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")
            action = parts[1] if len(parts) > 1 else None
            arg = parts[2] if len(parts) > 2 else None
            extra = parts[3] if len(parts) > 3 else None
            if action == "pick" and arg:
                name = arg
                text = f"Load Saved Image\n\nSelected: {name}"
                # Determine type by prefix
                is_bg = name.startswith("bg_") or name.startswith("txtbg_")
                is_composite = name.startswith("composite_")
                # Weather prefs for this chat
                wx = self.load_wx.get(chat_id, {"wbadge": False, "woverlay": False})
                is_off = not wx.get("wbadge") and not wx.get("woverlay")
                # Build keyboard depending on type
                kb = []
                # Weather row
                kb.append([{ "text": "Weather:", "callback_data": "load|noop" }])
                if is_bg:
                    kb.append([
                        {"text": f"Off {'✅' if is_off else ''}", "callback_data": f"load|w|off|{name}"},
                        {"text": f"Badge {'✅' if wx.get('wbadge') else ''}", "callback_data": f"load|w|badge|{name}"},
                        {"text": f"Overlay {'✅' if wx.get('woverlay') else ''}", "callback_data": f"load|w|overlay|{name}"},
                        {"text": "Options", "callback_data": "wx|open"},
                    ])
                else:
                    # Composite: allow Badge only (no overlay)
                    kb.append([
                        {"text": f"Off {'✅' if is_off else ''}", "callback_data": f"load|w|off|{name}"},
                        {"text": f"Badge {'✅' if wx.get('wbadge') else ''}", "callback_data": f"load|w|badge|{name}"},
                        {"text": "Options", "callback_data": "wx|open"},
                    ])
                # Primary actions
                # Preview full-width
                kb.append([
                    {"text": "👁 Preview", "callback_data": f"load|preview|{name}"},
                ])
                # Display Now + Use with Text on same row (composite: only Display Now)
                row = [{"text": "🖼 Display Now", "callback_data": f"load|display|{name}"}]
                if is_bg:
                    row.append({"text": "📝 Use with Text", "callback_data": f"load|usetxt|{name}"})
                kb.append(row)
                # Back/cancel
                kb.append([
                    {"text": "⬅️ Back", "callback_data": "load|back"},
                    {"text": "✖️ Cancel", "callback_data": "load|cancel"},
                ])
                markup = {"inline_keyboard": kb}
                self._refresh_load_message(chat_id, message_id, text, markup)
                self._answer_callback(callback_query["id"]) 
            elif action == "w" and arg:
                # arg in {off,badge,overlay}, extra holds the selected name
                name = extra
                if arg == "off":
                    self.load_wx[chat_id] = {"wbadge": False, "woverlay": False}
                elif arg == "badge":
                    self.load_wx[chat_id] = {"wbadge": True, "woverlay": False}
                elif arg == "overlay":
                    self.load_wx[chat_id] = {"wbadge": False, "woverlay": True}
                else:
                    self._answer_callback(callback_query["id"])
                    return
                # Answer this callback quickly to avoid timeout
                try:
                    label = "Off" if arg == "off" else ("Badge" if arg == "badge" else "Overlay")
                    self._answer_callback(callback_query["id"], text=f"Weather: {label}")
                except Exception:
                    logger.exception("Failed to answer weather selection callback")
                # Rebuild the picked UI to reflect selection
                self._handle_callback({"data": f"load|pick|{name}", "id": callback_query["id"], "message": callback_query["message"]})
                return
            elif action == "preview" and arg:
                name = arg
                path = os.path.join(self.text_flow.storage_dir, "saved", f"{name}.png")
                try:
                    with Image.open(path) as img:
                        final_img = img.convert("RGB")
                        wx = self.load_wx.get(chat_id, {"wbadge": False, "woverlay": False})
                        is_composite = name.startswith("composite_")
                        try:
                            if wx.get("wbadge"):
                                final_img = self.text_flow.overlay_weather_badge(final_img)
                            # Overlay allowed only if not composite
                            if wx.get("woverlay") and not is_composite:
                                final_img = self.text_flow.overlay_weather_caption(final_img)
                            else:
                                opts = self._get_weather_options()
                                if (opts.get("weather", {}).get("overlay", {}).get("enabled")) and not is_composite:
                                    final_img = self.text_flow.overlay_weather_caption(final_img)
                        except Exception:
                            logger.exception("Failed to apply weather overlays for load preview")
                        self._send_photo(chat_id, final_img, caption=f"Preview: {name}")
                    self._answer_callback(callback_query["id"], text="Preview sent.")
                except Exception as exc:
                    logger.exception("Load preview failed: %s", exc)
                    self._answer_callback(callback_query["id"], text="Preview failed.")
            elif action == "page" and arg is not None:
                try:
                    page = int(arg)
                except ValueError:
                    page = 0
                # Rebuild list on requested page by editing current message
                try:
                    names = self.text_flow._list_saved_names()  # pylint: disable=protected-access
                except Exception:
                    logger.exception("Failed to list saved images.")
                    names = []
                page_size = 6
                total_pages = max(1, (len(names) + page_size - 1) // page_size)
                page = max(0, min(page, total_pages - 1))
                start = page * page_size
                end = start + page_size
                kb = []
                row = []
                for name in names[start:end]:
                    base = self.text_flow._strip_prefix(name)  # pylint: disable=protected-access
                    label = base if len(base) <= 18 else (base[:15] + "…")
                    row.append({"text": label, "callback_data": f"load|pick|{name}"})
                    if len(row) == 3:
                        kb.append(row)
                        row = []
                if row:
                    kb.append(row)
                nav = []
                if total_pages > 1 and page > 0:
                    nav.append({"text": "◀️ Prev", "callback_data": f"load|page|{page-1}"})
                if total_pages > 1 and page < total_pages - 1:
                    nav.append({"text": "Next ▶️", "callback_data": f"load|page|{page+1}"})
                if nav:
                    kb.append(nav)
                last_bg = os.path.join(self.text_flow.storage_dir, "last_text_background.png")
                if os.path.exists(last_bg):
                    kb.append([{ "text": "🖼 Display Last Background (/txt)", "callback_data": "load|display_last" }])
                kb.append([{ "text": "✖️ Cancel", "callback_data": "load|cancel" }])
                self._refresh_load_message(chat_id, message_id, "Load Saved Image\n\nPick an image to preview, display, or use as background.", {"inline_keyboard": kb})
                self._answer_callback(callback_query["id"], text=f"Page {page+1}")
            elif action == "filter" and arg in {"all", "bg", "composite"}:
                # Update filter and refresh list on page 0
                self.load_filters[chat_id] = arg
                try:
                    names = self.text_flow._list_saved_names()  # pylint: disable=protected-access
                except Exception:
                    logger.exception("Failed to list saved images.")
                    names = []
                if arg == "bg":
                    names = [n for n in names if n.startswith("bg_") or n.startswith("txtbg_")]
                elif arg == "composite":
                    names = [n for n in names if n.startswith("composite_")]
                page_size = 6
                start = 0
                end = min(len(names), page_size)
                kb = []
                # Filter row
                def filt_btn3(key, label):
                    sel = self.load_filters.get(chat_id, "all") == key
                    txt = f"{label} {'✅' if sel else ''}".strip()
                    return {"text": txt, "callback_data": f"load|filter|{key}"}
                kb.append([filt_btn3("all", "All"), filt_btn3("bg", "Backgrounds"), filt_btn3("composite", "Composites")])
                # Items
                row = []
                for name in names[start:end]:
                    base = self.text_flow._strip_prefix(name)  # pylint: disable=protected-access
                    label = base if len(base) <= 18 else (base[:15] + "…")
                    row.append({"text": label, "callback_data": f"load|pick|{name}"})
                    if len(row) == 3:
                        kb.append(row)
                        row = []
                if row:
                    kb.append(row)
                # Next link if more pages
                total_pages = max(1, (len(names) + page_size - 1) // page_size)
                if total_pages > 1:
                    kb.append([{ "text": "Next ▶️", "callback_data": "load|page|1" }])
                last_bg = os.path.join(self.text_flow.storage_dir, "last_text_background.png")
                if os.path.exists(last_bg):
                    kb.append([{ "text": "🖼 Display Last Background (/txt)", "callback_data": "load|display_last" }])
                kb.append([{ "text": "✖️ Cancel", "callback_data": "load|cancel" }])
                self._refresh_load_message(chat_id, message_id, "Load Saved Image\n\nPick an image to preview, display, or use as background.", {"inline_keyboard": kb})
                self._answer_callback(callback_query["id"], text=f"Filter: {arg}")
            elif action == "display" and arg:
                name = arg
                path = os.path.join(self.text_flow.storage_dir, "saved", f"{name}.png")
                try:
                    # Build status details
                    wx = self.load_wx.get(chat_id, {"wbadge": False, "woverlay": False})
                    is_composite = name.startswith("composite_")
                    is_bg = name.startswith("bg_") or name.startswith("txtbg_")
                    badge_pos = self._get_weather_options().get("weather", {}).get("badge", {}).get("position", "TR").upper()
                    if wx.get("wbadge"):
                        wlabel = f"Badge ({badge_pos})"
                    elif wx.get("woverlay") and not is_composite:
                        wlabel = "Overlay"
                    else:
                        # Global overlay (if any) applies later; show Off here for clarity
                        wlabel = "Off"
                    # Provider/units
                    prov_label = "-"
                    units = "-"
                    try:
                        plugin, settings = self.text_flow._get_weather_plugin_and_settings()  # pylint: disable=protected-access
                        if settings:
                            provider = (settings.get("weatherProvider") or "OpenWeatherMap").strip()
                            prov_label = "OWM" if provider == "OpenWeatherMap" else ("OM" if provider == "OpenMeteo" else provider)
                            units = (settings.get("units") or "metric").strip()
                    except Exception:
                        logger.exception("Failed to read weather provider/units for status")
                    status_lines = [
                        f"Displaying '{name}'…",
                        f"Type: {'Background' if is_bg else 'Composite'}",
                        f"Weather: {wlabel} | Provider: {prov_label} | Units: {units}",
                    ]
                    status_text = "\n".join(status_lines)
                    # Answer callback immediately (avoid 10s timeout) and lock UI
                    try:
                        self._answer_callback(callback_query["id"], text="Displaying…")
                    except Exception:
                        logger.exception("Failed to answer display callback quickly")
                    try:
                        self._refresh_load_message(chat_id, message_id, status_text, {"inline_keyboard": []})
                    except Exception:
                        logger.exception("Failed to set displaying status for /load")
                    with Image.open(path) as img:
                        final_img = img.convert("RGB")
                        try:
                            if wx.get("wbadge"):
                                final_img = self.text_flow.overlay_weather_badge(final_img)
                            if wx.get("woverlay") and not is_composite:
                                final_img = self.text_flow.overlay_weather_caption(final_img)
                            else:
                                opts = self._get_weather_options()
                                if (opts.get("weather", {}).get("overlay", {}).get("enabled")) and not is_composite:
                                    final_img = self.text_flow.overlay_weather_caption(final_img)
                        except Exception:
                            logger.exception("Failed to apply weather overlays for load display")
                        self.display_manager.display_image(final_img)
                    # Send the displayed image back to chat
                    try:
                        self._send_photo(chat_id, final_img, caption=f"Displayed: {name}")
                    except Exception:
                        logger.exception("Failed to send displayed image back to Telegram for /load")
                    # Update UI to indicate completion
                    done_lines = [
                        f"Loaded '{name}' to display.",
                        f"Type: {'Background' if is_bg else 'Composite'}",
                        f"Weather: {wlabel} | Provider: {prov_label} | Units: {units}",
                    ]
                    self._refresh_load_message(chat_id, message_id, "\n".join(done_lines), {"inline_keyboard": []})
                except Exception as exc:
                    logger.exception("Load display failed: %s", exc)
                    try:
                        self._answer_callback(callback_query["id"], text="Load failed.")
                    except Exception:
                        logger.exception("Failed to answer callback after load failure")
            elif action == "display_last":
                path = os.path.join(self.text_flow.storage_dir, "last_text_background.png")
                if not os.path.exists(path):
                    self._answer_callback(callback_query["id"], text="No last /txt background.")
                    return
                try:
                    with Image.open(path) as img:
                        self.display_manager.display_image(img)
                    self._refresh_load_message(chat_id, message_id, "Displayed last /txt background.", {"inline_keyboard": []})
                    self._answer_callback(callback_query["id"], text="Displayed.")
                except Exception as exc:
                    logger.exception("Display last background failed: %s", exc)
                    self._answer_callback(callback_query["id"], text="Load failed.")
            elif action == "usetxt" and arg:
                name = arg
                self.pending_txt_bg[chat_id] = name
                try:
                    self._api_post(
                        "sendMessage",
                        data={
                            "chat_id": chat_id,
                            "text": f"Enter the message text to render over '{name}':",
                            "reply_markup": json.dumps({"force_reply": True, "input_field_placeholder": "Type your message"}),
                        },
                    )
                except Exception:
                    logger.exception("Failed to prompt for /txt message after load.")
                self._answer_callback(callback_query["id"], text="Awaiting message…")
            # removed bgonly action (Display Now is the default no-text path)
            elif action == "back":
                self._refresh_load_message(chat_id, message_id, "Load Saved Image\n\nPick an image to preview, display, or use as background.", {
                    "inline_keyboard": [[{"text": "✖️ Cancel", "callback_data": "load|cancel"}]]
                })
                # Reopen the initial menu with first page
                self._send_load_menu(chat_id)
            elif action == "cancel":
                self._refresh_load_message(chat_id, message_id, "Load cancelled.", {"inline_keyboard": []})
                self._answer_callback(callback_query["id"], text="Cancelled.")
            else:
                self._answer_callback(callback_query["id"]) 
        else:
            self._answer_callback(callback_query["id"])

    def _init_ai_prompt(self, chat_id, prompt, source_text_request_id=None):
        prompt = prompt.strip()
        if not prompt:
            self._send_message(chat_id, "Prompt cannot be empty.")
            return

        request_id = f"{chat_id}:{int(time.time()*1000)}"
        model_values = [m[0] for m in self.AI_MODELS]
        fallback_model = self.AI_MODELS[0][0]
        model = self._resolve_default_model(
            self.device_config.load_env_key("TELEGRAM_AI_DEFAULT_MODEL"),
            model_values,
            fallback=fallback_model,
        )
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
            "source_text_request_id": source_text_request_id,
            "wbadge": False,
            "woverlay": False,
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

    def _init_text_prompt(self, chat_id, message):
        message = message.strip()
        if not message:
            self._send_message(chat_id, "Usage: /txt <message>")
            return

        request = self.text_flow.create_request(chat_id, message)
        summary = self.text_flow.format_summary(request)
        markup = self.text_flow.build_keyboard(request)
        response = self._api_post(
            "sendMessage",
            data={
                "chat_id": chat_id,
                "text": summary,
                "reply_markup": json.dumps(markup),
            },
        )
        self.text_flow.set_message_id(request["id"], response["result"]["message_id"])

    def _start_custom_background_flow(self, text_request):
        prompt = text_request.get("image_prompt") or text_request["text"].strip()
        if not prompt:
            prompt = "Background"
        self._init_ai_prompt(text_request["chat_id"], prompt, source_text_request_id=text_request["id"])

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

    def _get_palette_label(self, palette_key):
        return self.PALETTE_LABELS.get(palette_key, palette_key.replace("_", " ").title())

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
        palette_label = self._get_palette_label(request["palette"])
        header = "🎨 AI Image Prompt"
        if request.get("source_text_request_id"):
            header = "🖼 Background Image"
        lines = [
            header,
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

        keyboard = [
            [
                {"text": f"⚙️ Model: {model_label}", "callback_data": f"ai|{request_id}|cycle_model"},
                {"text": f"📐 Quality: {quality_text}", "callback_data": f"ai|{request_id}|cycle_quality"},
            ]
        ]

        for row in self.STYLE_ROWS:
            keyboard.append([self._style_button(request_id, request, style_value) for style_value in row])

        palette_label = self._get_palette_label(request["palette"])
        keyboard.append(
            [
                {
                    "text": f"🎨 Palette: {palette_label}",
                    "callback_data": f"ai|{request_id}|cycle_palette",
                }
            ]
        )
        # Weather selection row (Off/Badge/Overlay/Options)
        wbadge_on = bool(request.get("wbadge"))
        woverlay_on = bool(request.get("woverlay"))
        is_off = not wbadge_on and not woverlay_on
        keyboard.append(
            [
                {"text": f"Weather:", "callback_data": f"ai|{request_id}|noop"},
            ]
        )
        keyboard.append(
            [
                {"text": f"Off {'✅' if is_off else ''}".strip(), "callback_data": f"ai|{request_id}|woff"},
                {"text": f"Badge {'✅' if wbadge_on else ''}".strip(), "callback_data": f"ai|{request_id}|wbadge|on"},
                {"text": f"Overlay {'✅' if woverlay_on else ''}".strip(), "callback_data": f"ai|{request_id}|wover|on"},
                {"text": "Options", "callback_data": "wx|open"},
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

    # --- Weather options menu ----------------------------------------------

    def _get_weather_options(self):
        opts = self.device_config.get_config("telegram_options", default={}) or {}
        # Fill defaults
        opts.setdefault("weather", {})
        opts["weather"].setdefault("badge", {"enabled": False, "position": "tr"})
        opts["weather"].setdefault("overlay", {"enabled": False})
        return opts

    def _set_weather_options(self, opts):
        cfg = self.device_config.get_config()
        cfg["telegram_options"] = opts
        self.device_config.update_config(cfg)

    def _send_weather_menu(self, chat_id, message_id=None):
        opts = self._get_weather_options()
        badge_on = opts["weather"]["badge"]["enabled"]
        pos = opts["weather"]["badge"]["position"].upper()
        overlay_on = opts["weather"]["overlay"]["enabled"]
        lines = [
            "🌦 Weather Options",
            "",
            f"Badge: {'On' if badge_on else 'Off'} (pos {pos})",
            f"Overlay: {'On' if overlay_on else 'Off'}",
            "",
            "Notes:",
            "- Badge adds a small temperature icon in a corner.",
            "- Full overlay adds a bottom caption bar with weather summary.",
            "- When overlay is On here, it applies to AI, text, /load, and slideshows (skipped for composite images).",
        ]
        text = "\n".join(lines)

        kb = {
            "inline_keyboard": [
                [
                    {"text": f"🌤 Badge: {'On' if badge_on else 'Off'}", "callback_data": f"wx|badge|{'off' if badge_on else 'on'}"},
                ],
                [
                    {"text": f"📍 Badge Pos: {pos}", "callback_data": "wx|pos|cycle"},
                ],
                [
                    {"text": f"🗒️ Full Overlay: {'On' if overlay_on else 'Off'}", "callback_data": f"wx|overlay|{'off' if overlay_on else 'on'}"},
                ],
                [
                    {"text": "✖️ Close", "callback_data": "wx|close"},
                ],
            ]
        }

        if message_id:
            try:
                self._api_post("editMessageText", data={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": text,
                    "reply_markup": json.dumps(kb),
                })
            except Exception:
                logger.exception("Failed to edit weather menu; sending new one.")
                self._api_post("sendMessage", data={"chat_id": chat_id, "text": text, "reply_markup": json.dumps(kb)})
        else:
            resp = self._api_post("sendMessage", data={"chat_id": chat_id, "text": text, "reply_markup": json.dumps(kb)})
            try:
                mid = resp.get("result", {}).get("message_id")
                if mid:
                    self.wx_menu_ids[chat_id] = mid
            except Exception:
                pass

    def _refresh_weather_menu(self, chat_id, message_id=None):
        message_id = message_id or self.wx_menu_ids.get(chat_id)
        if message_id:
            self._send_weather_menu(chat_id, message_id)
        else:
            self._send_weather_menu(chat_id)

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

    def _refresh_text_message(self, request, status=None):
        summary = self.text_flow.format_summary(request, status=status)
        if request.get("locked"):
            markup = {"inline_keyboard": []}
        else:
            markup = self.text_flow.build_keyboard(request)
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
            send_photo_fn, finalize_fn, saved_path = self._generate_ai_image(request)
        except Exception as exc:
            logger.exception("AI generation failed: %s", exc)
            self._send_message(request["chat_id"], f"Failed to generate image: {exc}")
            source_text_id = request.get("source_text_request_id")
            if source_text_id:
                text_request = self.text_flow.get_request(source_text_id)
                if text_request:
                    text_request["locked"] = False
                    text_request["awaiting_background"] = False
                    text_request["custom_background"] = None
                    try:
                        self._refresh_text_message(text_request, status="Background failed. Adjust options.")
                    except Exception:
                        logger.exception("Failed to update text message after background error.")
            self._cancel_ai_request(request_id, status_text="Failed.")
            return

        source_text_id = request.get("source_text_request_id")
        if source_text_id:
            caption = "✅ Background image ready"
            threading.Thread(
                target=send_photo_fn,
                args=(caption,),
                name=f"TelegramPhotoBG-{request_id}",
                daemon=True,
            ).start()

            text_request = self.text_flow.attach_custom_background(source_text_id, saved_path)
            if not text_request:
                self._send_message(request["chat_id"], "Background saved, but original text request expired.")
            else:
                self._refresh_text_message(text_request, status="Background ready. Rendering…")
                threading.Thread(
                    target=self._process_text_request,
                    args=(source_text_id,),
                    name=f"TelegramTextFinalize-{source_text_id}",
                    daemon=True,
                ).start()

            self._cancel_ai_request(request_id, status_text="Background saved.")
            return

        caption = f"✅ Image generated with {dict(self.AI_MODELS)[request['model']]} ({request['quality']})"
        threading.Thread(
            target=send_photo_fn,
            args=(caption,),
            name=f"TelegramPhoto-{request_id}",
            daemon=True,
        ).start()

        try:
            finalize_fn()
        finally:
            self._cancel_ai_request(request_id, status_text="Completed.")

    def _process_text_request(self, request_id):
        request = self.text_flow.get_request(request_id)
        if not request:
            return

        try:
            # Precompute rewritten text (if enabled) and update summary before heavy work
            try:
                preview_text = self.text_flow.compute_final_text(request)
                request["final_text_preview"] = preview_text
                self._refresh_text_message(request, status="Rendering…")
            except Exception:
                logger.exception("Failed to compute preview text before rendering.")
            result = self.text_flow.finalize(request)
        except Exception as exc:
            logger.exception("Telegram text generation failed: %s", exc)
            request["locked"] = False
            self._refresh_text_message(request, status="Failed. Adjust options and try again.")
            self._send_message(request["chat_id"], f"Failed to render text: {exc}")
            return

        try:
            self._refresh_text_message(request, status="Completed.")
        except Exception:
            logger.exception("Failed to update Telegram message after text rendering.")

        image_path = result["image_path"]
        caption = "✅ Text updated"
        try:
            with Image.open(image_path) as img:
                self._send_photo(request["chat_id"], img, caption=caption)
        except Exception:
            logger.exception("Failed to send Telegram text preview.")
            self._send_message(request["chat_id"], "Text displayed, but failed to send preview image.")

        self.text_flow.cancel_request(request_id)

    def _cancel_ai_request(self, request_id, status_text="Cancelled."):
        request = self.pending_requests.pop(request_id, None)
        if not request:
            return

        source_text_id = request.get("source_text_request_id")
        if source_text_id:
            text_request = self.text_flow.get_request(source_text_id)
            if text_request:
                if status_text not in {"Background saved."}:
                    text_request["locked"] = False
                    text_request["awaiting_background"] = False
                    text_request["custom_background"] = None
                    try:
                        self._refresh_text_message(text_request, status=status_text)
                    except Exception:
                        logger.exception("Failed to update text message after cancelling background request.")

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
        elif style == "drawing":
            settings["styleHint"] = "drawing"
        elif style == "far_side":
            settings["styleHint"] = "far_side"

        image = plugin.generate_image(settings, self.device_config)

        # Apply optional Weather badge and overlay (per-request and global)
        final_img = image
        try:
            if request.get("wbadge"):
                final_img = self.text_flow.overlay_weather_badge(final_img)
            # Full overlay: per-request or global option
            if request.get("woverlay"):
                final_img = self.text_flow.overlay_weather_caption(final_img)
            opts = self._get_weather_options()
            if (opts.get("weather", {}).get("overlay", {}).get("enabled")):
                final_img = self.text_flow.overlay_weather_caption(final_img)
        except Exception:
            logger.exception("Failed to apply weather overlays for AI image.")

        saved_path = self._save_image(final_img)

        def send_photo(caption=None):
            try:
                with Image.open(saved_path) as img:
                    self._send_photo(request.get("chat_id"), img, caption=caption)
            except Exception:
                logger.exception("Failed to send photo back to Telegram")

        def finalize():
            # Display the same image used for preview to keep parity
            self.display_manager.display_image(final_img, image_settings=plugin_config.get("image_settings", []))
            current_dt = (
                self.refresh_task._get_current_datetime()
                if hasattr(self.refresh_task, "_get_current_datetime")
                else datetime.utcnow()
            )
            image_hash = compute_image_hash(final_img)
            refresh_info = RefreshInfo(
                refresh_type="Telegram AI",
                plugin_id="ai_image",
                refresh_time=current_dt.isoformat(),
                image_hash=image_hash,
            )
            self.device_config.refresh_info = refresh_info
            self.device_config.write_config()

        return send_photo, finalize, saved_path

    def _send_photo(self, chat_id, image, caption=None):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        files = {"photo": ("image.png", buffer, "image/png")}
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        self._api_post("sendPhoto", data=data, files=files)

    def _send_help(self, chat_id):
        model_labels = ", ".join(label for _, label in self.AI_MODELS)
        lines = [
            "InkyPi Telegram Controls",
            "",
            "Quick start:",
            "- Send a photo to set the display.",
            "- /ai a short prompt — generate an AI image (then tap Generate).",
            "- /txt a short note — generate a note (pick a background, then Render).",
            "- /status — send the current display preview.",
            "- /cat — regenerate today’s Daily Theme background (if configured).",
            "- /cat <prompt> — set a custom scene for the rest of today (auto-enhanced).",
            "- /cat clear — go back to the auto scene generator.",
            "- /theme — Daily Theme dashboard (illustration + cards).",
            "- /card1, /card2, /card3 — pick card slots directly (optional).",
            "",
            "AI image (/ai):",
            "- /ai <prompt> — opens the image generator with buttons for Model/Quality/Style/Palette.",
            f"- Models: {model_labels}. (Gemini requires `GEMINI_API_KEY`)",
            "- Default model: set `TELEGRAM_AI_DEFAULT_MODEL` in your `.env` (list supported; first entry wins).",
            "- Weather controls: toggle Off/Badge/Overlay and open Weather options.",
            "",
            "Text composer (/txt):",
            "- /txt <message> — compose a note with style and background options.",
            "- Backgrounds: None, Last Image, Live Weather, AI background, Solid Colour, Saved Image, Custom AI prompt.",
            "- Weather controls: toggle Off/Badge/Overlay and open Weather options.",
            "",
            "Saved images:",
            "- /save — save the last background or the last note+background composite.",
            "- /load — browse saved images (preview, display now, or use as /txt background).",
            "- /load <name> — display a saved image immediately.",
            "",
            "Weather (/weather or /wx):",
            "- Configure badge vs overlay and badge position.",
            "- Requires the Weather plugin to be configured (location + weather API key).",
            "",
            "Slideshow (/slideshow):",
            "- /slideshow — open controls (filter, interval, shuffle, weather).",
            "- /slideshow start [all|bg|composite] [interval=N] [shuffle=on|off] [weather=off|badge|overlay]",
            "- /slideshow stop (or /stop)",
        ]
        markup = {
            "inline_keyboard": [
                [
                    {"text": "🖼 Current", "callback_data": "help|status"},
                    {"text": "🤖 AI Image", "callback_data": "help|ai"},
                    {"text": "📝 Text", "callback_data": "help|txt"},
                ],
                [
                    {"text": "💾 Save", "callback_data": "help|save"},
                    {"text": "📂 Load", "callback_data": "help|load"},
                ],
                [
                    {"text": "🌦 Weather", "callback_data": "help|weather"},
                    {"text": "🎞 Slideshow", "callback_data": "help|slideshow"},
                ],
                [
                    {"text": "🎨 New Daily Theme", "callback_data": "help|cat"},
                    {"text": "🎨 Daily Theme", "callback_data": "help|theme"},
                ],
                [
                    {"text": "✖️ Close", "callback_data": "help|close"},
                ],
            ]
        }
        self._api_post(
            "sendMessage",
            data={
                "chat_id": chat_id,
                "text": "\n".join(lines),
                "reply_markup": json.dumps(markup),
            },
        )

    @staticmethod
    def _sanitize_cache_id(value):
        value = (value or "").strip().lower()
        value = re.sub(r"[^a-z0-9_-]+", "-", value)
        value = re.sub(r"-{2,}", "-", value).strip("-")
        return value or "default"

    @staticmethod
    def _parse_hhmm(value):
        value = (value or "").strip()
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
        if not match:
            return dt_time(4, 0)
        hour = max(0, min(23, int(match.group(1))))
        minute = max(0, min(59, int(match.group(2))))
        return dt_time(hour, minute)

    @staticmethod
    def _day_key(now, refresh_time):
        if now.timetz().replace(tzinfo=None) < refresh_time:
            day = (now - timedelta(days=1)).date()
        else:
            day = now.date()
        return day.isoformat()

    def _send_photo_path(self, chat_id, image_path, caption=None):
        if not image_path or not os.path.exists(image_path):
            self._send_message(chat_id, "No image available yet.")
            return
        with open(image_path, "rb") as img_file:
            files = {"photo": img_file}
            data = {"chat_id": chat_id}
            if caption:
                data["caption"] = caption
            self._api_post("sendPhoto", data=data, files=files)

    def _find_daily_cat_plugin_instance(self, current_dt):
        playlist_manager = self.device_config.get_playlist_manager()
        playlist = playlist_manager.determine_active_playlist(current_dt)
        if playlist:
            for plugin_instance in playlist.plugins:
                if plugin_instance.plugin_id == "daily_cat_weather":
                    return playlist, plugin_instance

        for playlist in playlist_manager.playlists:
            for plugin_instance in playlist.plugins:
                if plugin_instance.plugin_id == "daily_cat_weather":
                    return playlist, plugin_instance

        return None, None

    def _find_daily_theme_card_plugin_instance(self, current_dt, *, slot=1):
        def _matches_slot(instance_name):
            token = re.sub(r"[^a-z0-9]+", "", (instance_name or "").strip().lower())
            if token == f"card{slot}":
                return True
            if token == f"dailycard{slot}":
                return True
            if token == f"dailythemecard{slot}":
                return True
            if token.startswith("card") and token.endswith(str(slot)):
                return True
            return False

        playlist_manager = self.device_config.get_playlist_manager()
        playlist = playlist_manager.determine_active_playlist(current_dt)
        if playlist:
            candidates = [p for p in playlist.plugins if p.plugin_id == "daily_theme_card"]
            for plugin_instance in candidates:
                if _matches_slot(getattr(plugin_instance, "name", "")):
                    return playlist, plugin_instance
            if candidates and int(slot) == 1:
                return playlist, candidates[0]

        for playlist in playlist_manager.playlists:
            candidates = [p for p in playlist.plugins if p.plugin_id == "daily_theme_card"]
            for plugin_instance in candidates:
                if _matches_slot(getattr(plugin_instance, "name", "")):
                    return playlist, plugin_instance
            if candidates and int(slot) == 1:
                return playlist, candidates[0]

        return None, None

    def _daily_card_bg_label(self, mode: str) -> str:
        mode = (mode or "").strip().lower()
        labels = {
            "plain": "Plain",
            "illustration": "Illustration",
            "illustration_blur": "Illustration (Blur)",
        }
        return labels.get(mode, labels["plain"])

    def _send_daily_card_background_menu(
        self,
        chat_id,
        *,
        slot=1,
        message_id=None,
        include_back=False,
        status_line="",
        selected_mode=None,
    ):
        current_dt = (
            self.refresh_task._get_current_datetime()
            if hasattr(self.refresh_task, "_get_current_datetime")
            else datetime.utcnow()
        )
        playlist, plugin_instance = self._find_daily_theme_card_plugin_instance(current_dt, slot=slot)
        if not playlist or not plugin_instance:
            self._send_message(
                chat_id,
                f"Card {slot} isn't enabled yet. Add a `daily_theme_card` instance named `card{slot}` to a playlist.",
            )
            return

        settings = plugin_instance.settings or {}
        current_mode = (settings.get("backgroundMode") or "plain").strip().lower()
        if current_mode not in {"plain", "illustration", "illustration_blur"}:
            current_mode = "plain"
        selected_mode = (selected_mode or current_mode).strip().lower()
        if selected_mode not in {"plain", "illustration", "illustration_blur"}:
            selected_mode = current_mode

        lines = [
            f"Card {slot} Background",
            f"Current: {self._daily_card_bg_label(current_mode)} (`{current_mode}`)",
        ]
        if selected_mode != current_mode:
            lines.append(f"Selected: {self._daily_card_bg_label(selected_mode)} (`{selected_mode}`)")
        if status_line:
            lines.extend(["", status_line])
        lines.append("")
        lines.append("Choose a background:")

        def btn(mode: str):
            label = self._daily_card_bg_label(mode)
            text = f"✅ {label}" if mode == selected_mode else label
            return {"text": text, "callback_data": f"cardbg|{slot}|set|{mode}"}

        buttons = [
            [btn("plain"), btn("illustration")],
            [btn("illustration_blur")],
        ]
        if include_back:
            buttons.append(
                [
                    {"text": "⬅️ Back", "callback_data": f"daily|open|card|{slot}"},
                    {"text": "✖️ Close", "callback_data": "help|close"},
                ]
            )
        else:
            buttons.append([{"text": "✖️ Close", "callback_data": "help|close"}])

        payload = {"chat_id": chat_id, "text": "\n".join(lines), "reply_markup": json.dumps({"inline_keyboard": buttons})}
        if message_id:
            payload["message_id"] = message_id
            try:
                self._api_post("editMessageText", data=payload)
            except Exception as exc:
                if "message is not modified" not in str(exc).lower():
                    logger.exception("Failed to edit Card %s background menu.", slot)
        else:
            self._api_post("sendMessage", data=payload)

    def _set_daily_card_background(self, chat_id, *, slot=1, mode="", menu_message_id=None, include_back=False):
        mode = (mode or "").strip().lower()
        if mode not in {"plain", "illustration", "illustration_blur"}:
            self._send_message(chat_id, "Invalid background mode.")
            return

        if chat_id in self.pending_card_bg_updates:
            self._send_message(chat_id, f"Card {slot} background is already updating…")
            return

        self.pending_card_bg_updates.add(chat_id)
        if menu_message_id:
            self._send_daily_card_background_menu(
                chat_id,
                slot=slot,
                message_id=menu_message_id,
                include_back=include_back,
                status_line="Updating…",
                selected_mode=mode,
            )
        else:
            self._send_message(chat_id, f"Updating Card {slot} background…")

        threading.Thread(
            target=self._set_daily_card_background_worker,
            args=(chat_id, slot, mode, menu_message_id, include_back),
            name=f"TelegramCardBg{slot}-{chat_id}",
            daemon=True,
        ).start()

    def _set_daily_card_background_worker(self, chat_id, slot, mode, menu_message_id=None, include_back=False):
        try:
            current_dt = (
                self.refresh_task._get_current_datetime()
                if hasattr(self.refresh_task, "_get_current_datetime")
                else datetime.utcnow()
            )
            playlist, plugin_instance = self._find_daily_theme_card_plugin_instance(current_dt, slot=slot)
            if not playlist or not plugin_instance:
                self._send_message(
                    chat_id,
                    f"Card {slot} isn't enabled yet. Add a `daily_theme_card` instance named `card{slot}` to a playlist.",
                )
                return

            plugin_instance.settings = plugin_instance.settings or {}
            plugin_instance.settings["backgroundMode"] = mode
            try:
                self.device_config.write_config()
            except Exception:
                logger.exception("Failed to persist Card %s background mode.", slot)

            if getattr(self.refresh_task, "running", False):
                self.refresh_task.manual_update(PlaylistRefresh(playlist, plugin_instance, force=True))

            try:
                plugin_image_path = os.path.join(self.device_config.plugin_image_dir, plugin_instance.get_image_path())
                caption = f"🖼 Card {slot} background set to {self._daily_card_bg_label(mode)}"
                self._send_photo_path(chat_id, plugin_image_path, caption=caption)
            except Exception:
                logger.exception("Failed to send updated card image.")

            if menu_message_id:
                self._send_daily_card_background_menu(
                    chat_id,
                    slot=slot,
                    message_id=menu_message_id,
                    include_back=include_back,
                    status_line="Updated ✅",
                    selected_mode=mode,
                )
        except Exception as exc:
            logger.exception("Card %s background update failed: %s", slot, exc)
            self._send_message(chat_id, f"Card {slot} background update failed: {exc}")
        finally:
            try:
                self.pending_card_bg_updates.discard(chat_id)
            except Exception:
                pass

    def _clear_daily_cat_cache(self, plugin_instance, current_dt):
        settings = plugin_instance.settings or {}
        cache_id = self._sanitize_cache_id(settings.get("cacheId") or "default")
        daily_refresh_time = self._parse_hhmm(settings.get("dailyRefreshTime") or "04:00")
        day_key = self._day_key(current_dt, daily_refresh_time)

        cache_dir = os.path.join(self.device_config.BASE_DIR, "..", "mock_display_output", "daily_cat_weather")
        os.makedirs(cache_dir, exist_ok=True)

        targets = [
            os.path.join(cache_dir, f"bg_{cache_id}_{day_key}.png"),
            os.path.join(cache_dir, f"bg_{cache_id}_{day_key}.json"),
            os.path.join(cache_dir, f"latest_bg_{cache_id}.png"),
        ]

        removed = []
        for path in targets:
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
                removed.append(os.path.basename(path))
            except Exception:
                logger.exception("Failed to remove Daily Theme cache file: %s", path)

        return cache_id, day_key, removed

    def _get_openrouter_api_key(self):
        for key_name in ("OPEN_ROUTER_SECRET", "OPEN_ROUTER_API_KEY", "OPENROUTER_API_KEY"):
            value = (self.device_config.load_env_key(key_name) or "").strip()
            if value:
                return value
        return ""

    def _build_openrouter_prompt_client(self):
        api_key = self._get_openrouter_api_key()
        if not api_key:
            return None

        try:
            from plugins.ai_image.ai_image import AIImage  # local import to keep startup light
        except Exception:
            logger.exception("Failed to import AIImage helper; prompt enhancement disabled.")
            return None

        return {
            "type": "openrouter",
            "api_key": api_key,
            "model": AIImage._resolve_openrouter_model(self.device_config.load_env_key("OPEN_ROUTER_MODEL")),
            "referer": self.device_config.load_env_key("OPEN_ROUTER_REFERRER") or "https://github.com/fatihak/InkyPi",
            "title": self.device_config.load_env_key("OPEN_ROUTER_TITLE") or "InkyPi",
        }

    def _enhance_daily_cat_user_prompt(self, user_prompt, theme_id=None):
        if not user_prompt or not user_prompt.strip():
            return user_prompt

        prompt_client = self._build_openrouter_prompt_client()
        if not prompt_client:
            return user_prompt

        try:
            from plugins.ai_image.ai_image import AIImage  # local import for consistency
        except Exception:
            logger.exception("Failed to import AIImage helper; prompt enhancement disabled.")
            return user_prompt

        theme_prompt = ""
        character_prompt = ""
        try:
            from plugins.daily_cat_weather.daily_cat_weather import DailyCatWeather  # local import

            theme_spec = DailyCatWeather._theme_spec(theme_id)
            theme_prompt, character_prompt, _ = DailyCatWeather._theme_prompt_parts(theme_spec)
        except Exception:
            theme_prompt = ""
            character_prompt = ""

        system_content = (
            "You rewrite user ideas into a single strong image-generation prompt. "
            "Preserve the core idea and keep it playful and imaginative. Always include: "
            f"{character_prompt} "
            "No text, no captions, no speech bubbles, no letters, no numbers. Not photorealistic. Full-bleed scene. "
            "Do not create panels, split layouts, frames, borders, dividers, or any weather widgets (thermometers/gauges/icons). "
            "Add composition, setting, mood, and a few concrete visual details without introducing new main subjects. "
            f"{theme_prompt} "
            "Follow the PRIMARY art direction above strictly (do not default to a generic style). "
            "Keep under 60 words. Return only the refined prompt."
        )
        user_content = f"User idea: \"{user_prompt.strip()}\""

        try:
            refined = AIImage._call_prompt_service(prompt_client, system_content, user_content, temperature=0.8)
            refined = (refined or user_prompt).strip()
            logger.info(
                "Daily Theme prompt enhanced | raw=%s | enhanced=%s",
                self._trim_text(user_prompt, 200),
                self._trim_text(refined, 260),
            )
            return refined
        except Exception:
            logger.exception("Daily Theme prompt enhancement failed; using raw prompt.")
            return user_prompt

    def _daily_cat_theme_presets(self):
        try:
            from plugins.daily_cat_weather.daily_cat_weather import DailyCatWeather  # local import

            catalog = DailyCatWeather._theme_catalog()
            return {theme_id: {"label": spec.get("label") or theme_id} for theme_id, spec in (catalog or {}).items()}
        except Exception:
            return {}

    def _daily_card_presets(self):
        try:
            from plugins.daily_theme_card.daily_theme_card import DailyThemeCard  # local import

            catalog = DailyThemeCard._card_catalog()
            return {card_id: {"label": spec.get("label") or card_id} for card_id, spec in (catalog or {}).items()}
        except Exception:
            return {}

    def _daily_card_leaf_choices(self):
        try:
            from plugins.daily_theme_card.daily_theme_card import DailyThemeCard  # local import

            return DailyThemeCard._card_leaf_choices() or []
        except Exception:
            return []

    def _daily_card_groups(self):
        try:
            from plugins.daily_theme_card.daily_theme_card import DailyThemeCard  # local import

            return DailyThemeCard._card_groups() or []
        except Exception:
            return []

    def _send_daily_theme_dashboard(self, chat_id, *, message_id=None):
        current_dt = (
            self.refresh_task._get_current_datetime()
            if hasattr(self.refresh_task, "_get_current_datetime")
            else datetime.utcnow()
        )

        theme_line = "Illustration: not configured"
        try:
            _, cat_instance = self._find_daily_cat_plugin_instance(current_dt)
            if cat_instance:
                theme_id = ((cat_instance.settings or {}).get("imageTheme") or "storybook").strip().lower()
                presets = self._daily_cat_theme_presets()
                label = (presets.get(theme_id, {}).get("label") if presets else None) or theme_id
                theme_line = f"Illustration: {label} (`{theme_id}`)"
            else:
                theme_line = "Illustration: not configured (add `daily_cat_weather` to a playlist)"
        except Exception:
            theme_line = "Illustration: unavailable"

        card_lines = []
        for slot in (1, 2, 3):
            line = f"Card {slot}: not configured"
            try:
                _, card_instance = self._find_daily_theme_card_plugin_instance(current_dt, slot=slot)
                if card_instance:
                    card_id = ((card_instance.settings or {}).get("cardId") or "inspiration").strip().lower()
                    bg_mode = ((card_instance.settings or {}).get("backgroundMode") or "plain").strip().lower()
                    bg_label = self._daily_card_bg_label(bg_mode)
                    presets = self._daily_card_presets()
                    label = (presets.get(card_id, {}).get("label") if presets else None) or card_id
                    line = f"Card {slot}: {label} (`{card_id}`) • {bg_label}"
                else:
                    line = (
                        f"Card {slot}: not configured "
                        f"(add a `daily_theme_card` instance named `card{slot}` to a playlist)"
                    )
            except Exception:
                line = f"Card {slot}: unavailable"
            card_lines.append(line)

        text = "\n".join(
            [
                "Daily Theme",
                "",
                theme_line,
                *card_lines,
                "",
                "Use the buttons below to adjust.",
            ]
        )

        markup = {
            "inline_keyboard": [
                [
                    {"text": "🎨 Illustration Theme", "callback_data": "daily|open|illustration"},
                    {"text": "🗒️ Card 1", "callback_data": "daily|open|card|1"},
                ],
                [
                    {"text": "🗒️ Card 2", "callback_data": "daily|open|card|2"},
                    {"text": "🗒️ Card 3", "callback_data": "daily|open|card|3"},
                ],
                [
                    {"text": "🔄 Reroll", "callback_data": "help|cat"},
                    {"text": "🖼 Current", "callback_data": "help|status"},
                ],
                [
                    {"text": "✖️ Close", "callback_data": "help|close"},
                ],
            ]
        }

        payload = {"chat_id": chat_id, "text": text, "reply_markup": json.dumps(markup)}
        if message_id:
            payload["message_id"] = message_id
            try:
                self._api_post("editMessageText", data=payload)
            except Exception as exc:
                if "message is not modified" not in str(exc).lower():
                    logger.exception("Failed to edit Daily Theme dashboard: %s", exc)
        else:
            self._api_post("sendMessage", data=payload)

    def _resolve_daily_card_id(self, raw, *, slot=1):
        raw = (raw or "").strip()
        if not raw:
            return ""

        presets = self._daily_card_presets()
        if not presets:
            return ""

        keys = list(presets.keys())

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
            return ""

        token = self._normalize_theme_token(raw)
        for card_id in keys:
            if token == self._normalize_theme_token(card_id):
                return card_id

        for card_id, preset in presets.items():
            label = preset.get("label") or ""
            if token == self._normalize_theme_token(label):
                return card_id

        matches = []
        for card_id, preset in presets.items():
            label = preset.get("label") or ""
            if self._normalize_theme_token(card_id).startswith(token) or self._normalize_theme_token(label).startswith(token):
                matches.append(card_id)

        return matches[0] if len(matches) == 1 else ""

    def _send_daily_card_menu(
        self,
        chat_id,
        *,
        slot=1,
        message_id=None,
        include_back=False,
        status_line="",
        selected_card_id=None,
        group_id="",
    ):
        current_dt = (
            self.refresh_task._get_current_datetime()
            if hasattr(self.refresh_task, "_get_current_datetime")
            else datetime.utcnow()
        )
        playlist, plugin_instance = self._find_daily_theme_card_plugin_instance(current_dt, slot=slot)
        if not playlist or not plugin_instance:
            self._send_message(
                chat_id,
                f"Card {slot} isn't enabled yet. Add the `daily_theme_card` plugin to a playlist in the web UI first.",
            )
            return

        presets = self._daily_card_presets()
        leaf_choices = self._daily_card_leaf_choices()
        groups = self._daily_card_groups()
        actual_card = ((plugin_instance.settings or {}).get("cardId") or "inspiration").strip().lower()
        if presets and actual_card not in presets:
            actual_card = next(iter(presets.keys()), "inspiration")

        selected_card = selected_card_id or actual_card
        selected_card = (selected_card or "").strip().lower()
        if presets and selected_card not in presets:
            selected_card = actual_card

        actual_label = (presets.get(actual_card, {}).get("label") if presets else None) or actual_card
        selected_label = (presets.get(selected_card, {}).get("label") if presets else None) or selected_card

        group_id = (group_id or "").strip().lower()
        if group_id == "root":
            group_id = ""

        group = None
        if group_id:
            for g in groups:
                if (g.get("id") or "").strip().lower() == group_id:
                    group = g
                    break
            if not group:
                group_id = ""

        lines = [
            f"Card {slot}",
            f"Current: {actual_label} ({actual_card})",
        ]
        if selected_card_id and selected_card != actual_card:
            lines.append(f"Selected: {selected_label} ({selected_card})")
        if status_line:
            lines.extend(["", status_line])
        if group:
            lines.extend(["", f"{group.get('label') or group_id}", "", "Pick a category:"])
        else:
            lines.extend(["", f"Pick a card (or run `/card{slot} <name>`):"])
            for card in leaf_choices:
                cid = (card.get("id") or "").strip().lower()
                label = card.get("label") or cid
                if cid:
                    lines.append(f"- {label} — `{cid}`")
            for g in groups:
                gid = (g.get("id") or "").strip().lower()
                label = g.get("label") or gid
                if gid:
                    lines.append(f"- {label} — browse with buttons")

        buttons = []
        if group:
            row = []
            for child in group.get("children") or []:
                cid = (child.get("id") or "").strip().lower()
                label = child.get("label") or cid
                text_btn = f"✅ {label}" if cid == selected_card else label
                row.append({"text": text_btn, "callback_data": f"card|{slot}|set|{cid}"})
                if len(row) == 2:
                    buttons.append(row)
                    row = []
            if row:
                buttons.append(row)

            if include_back:
                buttons.append(
                    [
                        {"text": "⬅️ Cards", "callback_data": f"daily|open|card|{slot}"},
                        {"text": "⬅️ Back", "callback_data": "daily|open|dashboard"},
                    ]
                )
            else:
                buttons.append([{"text": "⬅️ Cards", "callback_data": f"daily|open|card|{slot}"}])
            buttons.append([{"text": "✖️ Close", "callback_data": "help|close"}])
        else:
            row = []
            for card in leaf_choices:
                cid = (card.get("id") or "").strip().lower()
                label = card.get("label") or cid
                if not cid:
                    continue
                text_btn = f"✅ {label}" if cid == selected_card else label
                row.append({"text": text_btn, "callback_data": f"card|{slot}|set|{cid}"})
                if len(row) == 2:
                    buttons.append(row)
                    row = []
            if row:
                buttons.append(row)

            selected_group = ""
            try:
                from plugins.daily_theme_card.daily_theme_card import DailyThemeCard  # local import

                selected_group = DailyThemeCard._card_group_for_id(selected_card)  # pylint: disable=protected-access
            except Exception:
                selected_group = ""

            for g in groups:
                gid = (g.get("id") or "").strip().lower()
                if not gid:
                    continue
                label = g.get("label") or gid
                text_btn = f"✅ {label}" if gid == selected_group else label
                buttons.append(
                    [
                        {
                            "text": f"▶️ {text_btn}",
                            "callback_data": f"daily|open|card|{slot}|group|{gid}",
                        }
                    ]
                )

            buttons.append([{"text": "🖼 Background", "callback_data": f"daily|open|cardbg|{slot}"}])

            if include_back:
                buttons.append(
                    [
                        {"text": "⬅️ Back", "callback_data": "daily|open|dashboard"},
                        {"text": "✖️ Close", "callback_data": "help|close"},
                    ]
                )
            else:
                buttons.append([{"text": "✖️ Close", "callback_data": "help|close"}])

        payload = {"chat_id": chat_id, "text": "\n".join(lines), "reply_markup": json.dumps({"inline_keyboard": buttons})}
        if message_id:
            payload["message_id"] = message_id
            try:
                self._api_post("editMessageText", data=payload)
            except Exception as exc:
                if "message is not modified" not in str(exc).lower():
                    logger.exception("Failed to edit Card %s menu.", slot)
        else:
            self._api_post("sendMessage", data=payload)

    def _set_daily_card(self, chat_id, *, slot=1, raw_card="", menu_message_id=None, include_back=False):
        raw_card = (raw_card or "").strip()
        if not raw_card or raw_card.lower() in {"help", "menu", "list", "cards", "?"}:
            self._send_daily_card_menu(chat_id, slot=slot, message_id=menu_message_id, include_back=include_back)
            return

        card_id = self._resolve_daily_card_id(raw_card, slot=slot)
        if not card_id:
            self._send_message(chat_id, f"Unknown card.\n\nTry `/card{slot}` to browse.")
            return

        if chat_id in self.pending_card_updates:
            self._send_message(chat_id, f"Card {slot} is already updating…")
            return

        self.pending_card_updates.add(chat_id)
        self._send_message(chat_id, f"Updating Card {slot}…")
        if menu_message_id:
            self._send_daily_card_menu(
                chat_id,
                slot=slot,
                message_id=menu_message_id,
                include_back=include_back,
                status_line="Updating…",
                selected_card_id=card_id,
            )
        threading.Thread(
            target=self._set_daily_card_worker,
            args=(chat_id, slot, card_id, menu_message_id, include_back),
            name=f"TelegramCard{slot}-{chat_id}",
            daemon=True,
        ).start()

    def _set_daily_card_worker(self, chat_id, slot, card_id, menu_message_id=None, include_back=False):
        try:
            current_dt = (
                self.refresh_task._get_current_datetime()
                if hasattr(self.refresh_task, "_get_current_datetime")
                else datetime.utcnow()
            )
            playlist, plugin_instance = self._find_daily_theme_card_plugin_instance(current_dt, slot=slot)
            if not playlist or not plugin_instance:
                self._send_message(
                    chat_id,
                    f"Card {slot} isn't enabled yet. Add the `daily_theme_card` plugin to a playlist in the web UI first.",
                )
                return

            plugin_instance.settings = plugin_instance.settings or {}
            plugin_instance.settings["cardId"] = card_id

            try:
                self.device_config.write_config()
            except Exception:
                logger.exception("Failed to persist Card %s settings.", slot)

            if not getattr(self.refresh_task, "running", False):
                self._send_message(chat_id, "Refresh task is not running; restart `inkypi.service` and try again.")
                return

            self.refresh_task.manual_update(PlaylistRefresh(playlist, plugin_instance, force=True))

            presets = self._daily_card_presets()
            label = (presets.get(card_id, {}).get("label") if presets else None) or card_id
            plugin_image_path = os.path.join(self.device_config.plugin_image_dir, plugin_instance.get_image_path())
            caption = f"🗒️ Card {slot} set to {label} (card={card_id})"
            self._send_photo_path(chat_id, plugin_image_path, caption=caption)

            if menu_message_id:
                self._send_daily_card_menu(
                    chat_id,
                    slot=slot,
                    message_id=menu_message_id,
                    include_back=include_back,
                    status_line="Updated ✅",
                    selected_card_id=card_id,
                )
        except Exception as exc:
            logger.exception("Card %s update failed: %s", slot, exc)
            self._send_message(chat_id, f"Card {slot} update failed: {exc}")
        finally:
            self.pending_card_updates.discard(chat_id)

    @staticmethod
    def _normalize_theme_token(value):
        return re.sub(r"[^a-z0-9]+", "", (value or "").strip().lower())

    def _resolve_daily_cat_theme_id(self, raw):
        raw = (raw or "").strip()
        if not raw:
            return ""

        presets = self._daily_cat_theme_presets()
        if not presets:
            return "storybook" if self._normalize_theme_token(raw) in {"storybook", "default"} else ""

        keys = list(presets.keys())

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
            return ""

        token = self._normalize_theme_token(raw)
        if token in {"random", "rand"}:
            return "random"

        for theme_id in keys:
            if token == self._normalize_theme_token(theme_id):
                return theme_id

        for theme_id, preset in presets.items():
            label = preset.get("label") or ""
            if token == self._normalize_theme_token(label):
                return theme_id

        matches = []
        for theme_id, preset in presets.items():
            label = preset.get("label") or ""
            if self._normalize_theme_token(theme_id).startswith(token) or self._normalize_theme_token(label).startswith(token):
                matches.append(theme_id)

        return matches[0] if len(matches) == 1 else ""

    def _send_daily_cat_theme_menu(self, chat_id, *, message_id=None, status_line="", selected_theme_id=None, include_back=False):
        current_dt = (
            self.refresh_task._get_current_datetime()
            if hasattr(self.refresh_task, "_get_current_datetime")
            else datetime.utcnow()
        )
        playlist, plugin_instance = self._find_daily_cat_plugin_instance(current_dt)
        if not playlist or not plugin_instance:
            self._send_message(
                chat_id,
                "Daily Theme isn't enabled yet. Add the `daily_cat_weather` plugin to a playlist in the web UI first.",
            )
            return

        presets = self._daily_cat_theme_presets()
        actual_theme = ((plugin_instance.settings or {}).get("imageTheme") or "storybook").strip().lower()
        if presets and actual_theme not in presets:
            actual_theme = "storybook"

        selected_theme = selected_theme_id or actual_theme
        selected_theme = (selected_theme or "").strip().lower()
        if presets and selected_theme not in presets:
            selected_theme = actual_theme

        actual_label = (presets.get(actual_theme, {}).get("label") if presets else None) or actual_theme
        selected_label = (presets.get(selected_theme, {}).get("label") if presets else None) or selected_theme

        lines = [
            "Daily Theme",
            f"Current: {actual_label} ({actual_theme})",
        ]
        if selected_theme_id and selected_theme != actual_theme:
            lines.append(f"Selected: {selected_label} ({selected_theme})")
        if status_line:
            lines.extend(["", status_line])
        lines.extend(["", "Pick a theme (or run `/theme <name>`):"])

        if presets:
            for idx, (theme_id, preset) in enumerate(presets.items(), start=1):
                label = preset.get("label") or theme_id
                lines.append(f"{idx}. {label} — `{theme_id}`")

        buttons = []
        theme_items = list(presets.items()) if presets else [("storybook", {"label": "Storybook"})]
        row = []
        for theme_id, preset in theme_items:
            label = preset.get("label") or theme_id
            text = f"✅ {label}" if theme_id == selected_theme else label
            row.append({"text": text, "callback_data": f"cat_theme|set|{theme_id}"})
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        buttons.append([{"text": "🎲 Random", "callback_data": "cat_theme|set|random"}])
        if include_back:
            buttons.append(
                [
                    {"text": "⬅️ Back", "callback_data": "daily|open|dashboard"},
                    {"text": "✖️ Close", "callback_data": "help|close"},
                ]
            )
        else:
            buttons.append([{"text": "✖️ Close", "callback_data": "help|close"}])

        payload = {"chat_id": chat_id, "text": "\n".join(lines), "reply_markup": json.dumps({"inline_keyboard": buttons})}
        if message_id:
            payload["message_id"] = message_id
            try:
                self._api_post("editMessageText", data=payload)
            except Exception as exc:
                # Telegram often returns "message is not modified" when the user taps the already-selected theme.
                if "message is not modified" not in str(exc).lower():
                    logger.exception("Failed to edit Daily Theme menu.")
        else:
            self._api_post("sendMessage", data=payload)

    def _set_daily_cat_theme(self, chat_id, raw_theme, menu_message_id=None, include_back=False):
        raw_theme = (raw_theme or "").strip()
        if not raw_theme or raw_theme.lower() in {"help", "menu", "list", "themes", "?"}:
            self._send_daily_cat_theme_menu(chat_id, message_id=menu_message_id, include_back=include_back)
            return

        theme_id = self._resolve_daily_cat_theme_id(raw_theme)
        if not theme_id:
            presets = self._daily_cat_theme_presets()
            choices = ", ".join(presets.keys()) if presets else "storybook"
            self._send_message(chat_id, f"Unknown theme. Choices: {choices}\n\nTry `/theme` to browse.")
            return

        if chat_id in self.pending_cat_reroll:
            self._send_message(chat_id, "Daily Theme is already updating…")
            return

        effective_theme_id = theme_id
        if theme_id == "random":
            presets = self._daily_cat_theme_presets()
            keys = list(presets.keys())
            effective_theme_id = random.choice(keys) if keys else "storybook"

        self.pending_cat_reroll.add(chat_id)
        self._send_message(chat_id, "Updating Daily Theme…")
        if menu_message_id:
            self._send_daily_cat_theme_menu(
                chat_id,
                message_id=menu_message_id,
                status_line="Updating…",
                selected_theme_id=effective_theme_id,
                include_back=include_back,
            )
        threading.Thread(
            target=self._set_daily_cat_theme_worker,
            args=(chat_id, effective_theme_id, menu_message_id, include_back),
            name=f"TelegramCatTheme-{chat_id}",
            daemon=True,
        ).start()

    def _set_daily_cat_theme_worker(self, chat_id, theme_id, menu_message_id=None, include_back=False):
        try:
            current_dt = (
                self.refresh_task._get_current_datetime()
                if hasattr(self.refresh_task, "_get_current_datetime")
                else datetime.utcnow()
            )
            playlist, plugin_instance = self._find_daily_cat_plugin_instance(current_dt)
            if not playlist or not plugin_instance:
                self._send_message(
                    chat_id,
                    "Daily Theme isn't enabled yet. Add the `daily_cat_weather` plugin to a playlist in the web UI first.",
                )
                return

            presets = self._daily_cat_theme_presets()

            plugin_instance.settings = plugin_instance.settings or {}
            plugin_instance.settings["imageTheme"] = theme_id

            try:
                current_nonce = int((plugin_instance.settings or {}).get("rerollNonce") or 0)
            except (TypeError, ValueError):
                current_nonce = 0
            plugin_instance.settings["rerollNonce"] = current_nonce + 1

            cache_id, day_key, removed = self._clear_daily_cat_cache(plugin_instance, current_dt)
            try:
                self.device_config.write_config()
            except Exception:
                logger.exception("Failed to persist Daily Theme settings.")

            if not getattr(self.refresh_task, "running", False):
                self._send_message(chat_id, "Refresh task is not running; restart `inkypi.service` and try again.")
                return

            self.refresh_task.manual_update(PlaylistRefresh(playlist, plugin_instance, force=True))

            plugin_image_path = os.path.join(self.device_config.plugin_image_dir, plugin_instance.get_image_path())
            label = (presets.get(theme_id, {}).get("label") if presets else None) or theme_id
            suffix = f"(theme={theme_id}, cacheId={cache_id}, day={day_key})"
            caption = f"🎨 Daily Theme set to {label} {suffix}"
            if not removed:
                caption += " (no cache files to clear)"
            self._send_photo_path(chat_id, plugin_image_path, caption=caption)
            if menu_message_id:
                self._send_daily_cat_theme_menu(
                    chat_id,
                    message_id=menu_message_id,
                    status_line="Updated ✅",
                    selected_theme_id=theme_id,
                    include_back=include_back,
                )
        except Exception as exc:
            logger.exception("Daily Theme update failed: %s", exc)
            self._send_message(chat_id, f"Daily Theme update failed: {exc}")
        finally:
            self.pending_cat_reroll.discard(chat_id)

    def _reroll_daily_cat(self, chat_id, user_prompt=None):
        if chat_id in self.pending_cat_reroll:
            self._send_message(chat_id, "Already refreshing the Daily Theme image…")
            return

        self.pending_cat_reroll.add(chat_id)
        if user_prompt:
            if user_prompt.strip().lower() in {"clear", "off", "auto", "default"}:
                self._send_message(chat_id, "Clearing custom Daily Theme prompt and regenerating…")
            else:
                self._send_message(chat_id, "Setting a custom Daily Theme prompt and regenerating…")
        else:
            self._send_message(chat_id, "Refreshing Daily Theme…")
        threading.Thread(
            target=self._reroll_daily_cat_worker,
            args=(chat_id, user_prompt),
            name=f"TelegramCatReroll-{chat_id}",
            daemon=True,
        ).start()

    def _reroll_daily_cat_worker(self, chat_id, user_prompt=None):
        try:
            current_dt = (
                self.refresh_task._get_current_datetime()
                if hasattr(self.refresh_task, "_get_current_datetime")
                else datetime.utcnow()
            )
            playlist, plugin_instance = self._find_daily_cat_plugin_instance(current_dt)
            if not playlist or not plugin_instance:
                self._send_message(
                    chat_id,
                    "Daily Theme isn't enabled yet. Add the `daily_cat_weather` plugin to a playlist in the web UI first.",
                )
                return

            # Bump a nonce so rerolls naturally vary even when the weather is similar.
            try:
                current_nonce = int((plugin_instance.settings or {}).get("rerollNonce") or 0)
            except (TypeError, ValueError):
                current_nonce = 0
            plugin_instance.settings = plugin_instance.settings or {}
            plugin_instance.settings["rerollNonce"] = current_nonce + 1

            # Optional: set a custom prompt for the rest of the day (based on the plugin's dailyRefreshTime rollover).
            if user_prompt and user_prompt.strip():
                cmd = user_prompt.strip()
                cmd_lower = cmd.lower()

                if cmd_lower in {"clear", "off", "auto", "default"}:
                    for key_name in ("customPrompt", "customPromptEnhanced", "customPromptDayKey"):
                        plugin_instance.settings.pop(key_name, None)
                else:
                    daily_refresh_time = self._parse_hhmm(plugin_instance.settings.get("dailyRefreshTime") or "04:00")
                    day_key = self._day_key(current_dt, daily_refresh_time)
                    enhanced = self._enhance_daily_cat_user_prompt(
                        cmd,
                        theme_id=(plugin_instance.settings or {}).get("imageTheme"),
                    )
                    plugin_instance.settings["customPrompt"] = cmd
                    plugin_instance.settings["customPromptEnhanced"] = enhanced
                    plugin_instance.settings["customPromptDayKey"] = day_key
                    self._send_message(
                        chat_id,
                        "Custom Daily Theme prompt set for today.\n\n"
                        f"Enhanced prompt:\n{enhanced}\n\n"
                        "To clear: `/cat clear`",
                    )

            cache_id, day_key, removed = self._clear_daily_cat_cache(plugin_instance, current_dt)
            try:
                self.device_config.write_config()
            except Exception:
                logger.exception("Failed to persist Daily Theme settings.")

            if not getattr(self.refresh_task, "running", False):
                self._send_message(chat_id, "Refresh task is not running; restart `inkypi.service` and try again.")
                return

            self.refresh_task.manual_update(PlaylistRefresh(playlist, plugin_instance, force=True))

            plugin_image_path = os.path.join(
                self.device_config.plugin_image_dir, plugin_instance.get_image_path()
            )

            mode = "custom" if (plugin_instance.settings.get("customPromptDayKey") == day_key and plugin_instance.settings.get("customPromptEnhanced")) else "auto"
            theme_id = ((plugin_instance.settings or {}).get("imageTheme") or "storybook").strip().lower()
            suffix = (
                f"(mode={mode}, theme={theme_id}, cacheId={cache_id}, day={day_key}, "
                f"reroll={plugin_instance.settings.get('rerollNonce')})"
            )
            if removed:
                caption = f"🎨 Daily Theme refreshed {suffix}"
            else:
                caption = f"🎨 Daily Theme refreshed {suffix} (no cache files to clear)"
            self._send_photo_path(chat_id, plugin_image_path, caption=caption)
        except Exception as exc:
            logger.exception("Daily Theme reroll failed: %s", exc)
            self._send_message(chat_id, f"Daily Theme refresh failed: {exc}")
        finally:
            self.pending_cat_reroll.discard(chat_id)

    def _send_save_menu(self, chat_id):
        text = (
            "Save Image\n\n"
            "Pick which image to save and optionally choose a name."
        )
        markup = {
            "inline_keyboard": [
                [
                    {"text": "Save Background (Auto)", "callback_data": "save|choose|bgauto"},
                    {"text": "Save Last Background and Text", "callback_data": "save|choose|text"},
                ],
                [
                    {"text": "✖️ Cancel", "callback_data": "save|cancel"},
                ],
            ]
        }
        self._api_post("sendMessage", data={"chat_id": chat_id, "text": text, "reply_markup": json.dumps(markup)})

    def _send_load_menu(self, chat_id):
        text = (
            "Load Saved Image\n\n"
            "Pick an image to preview, display, use as background, or include in slideshows started with /slideshow."
        )
        try:
            names = self.text_flow._list_saved_names()  # pylint: disable=protected-access
        except Exception:
            logger.exception("Failed to list saved images.")
            names = []

        # Apply current filter
        filt = self.load_filters.get(chat_id, "all")
        if filt == "bg":
            names = [n for n in names if n.startswith("bg_") or n.startswith("txtbg_")]
        elif filt == "composite":
            names = [n for n in names if n.startswith("composite_")]

        page_size = 6
        total_pages = max(1, (len(names) + page_size - 1) // page_size)
        start = 0
        end = min(len(names), page_size)

        kb = []
        # Filter row
        def filt_btn(key, label):
            sel = self.load_filters.get(chat_id, "all") == key
            txt = f"{label} {'✅' if sel else ''}".strip()
            return {"text": txt, "callback_data": f"load|filter|{key}"}
        kb.append([filt_btn("all", "All"), filt_btn("bg", "Backgrounds"), filt_btn("composite", "Composites")])

        # Saved items (3 per row)
        row = []
        for name in names[start:end]:
            base = self.text_flow._strip_prefix(name)  # pylint: disable=protected-access
            label = base if len(base) <= 18 else (base[:15] + "…")
            row.append({"text": label, "callback_data": f"load|pick|{name}"})
            if len(row) == 3:
                kb.append(row)
                row = []
        if row:
            kb.append(row)

        # Nav row (Next only from first page)
        if total_pages > 1:
            kb.append([{ "text": "Next ▶️", "callback_data": "load|page|1" }])

        # Quick action: display last /txt background if available
        last_bg = os.path.join(self.text_flow.storage_dir, "last_text_background.png")
        if os.path.exists(last_bg):
            kb.append([{ "text": "🖼 Display Last Background (/txt)", "callback_data": "load|display_last" }])

        kb.append([{ "text": "✖️ Cancel", "callback_data": "load|cancel" }])

        self._api_post(
            "sendMessage",
            data={
                "chat_id": chat_id,
                "text": text,
                "reply_markup": json.dumps({"inline_keyboard": kb}),
            },
        )

    def _refresh_load_message(self, chat_id, message_id, text, markup):
        try:
            self._api_post(
                "editMessageText",
                data={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": text,
                    "reply_markup": json.dumps(markup) if markup else json.dumps({"inline_keyboard": []}),
                },
            )
        except Exception:
            logger.exception("Failed to update /load message.")

    def _suggest_save_name(self, source):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if source == "text":
            prefix = "composite"
        elif source == "txtbg":
            prefix = "txtbg"
        else:
            prefix = "bg"
        return f"{prefix}_{ts}"

    def _refresh_save_message(self, chat_id, message_id, text, markup):
        try:
            self._api_post(
                "editMessageText",
                data={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": text,
                    "reply_markup": json.dumps(markup) if markup else json.dumps({"inline_keyboard": []}),
                },
            )
        except Exception:
            logger.exception("Failed to update /save message.")

    def _answer_callback(self, callback_id, text=None, alert=False):
        data = {"callback_query_id": callback_id}
        if text:
            data["text"] = text
        if alert:
            data["show_alert"] = True
        self._api_post("answerCallbackQuery", data=data)

    # --- Slideshow ----------------------------------------------------------

    def _handle_slideshow_command(self, text, chat_id):
        parts = text.split()
        if len(parts) == 1:
            # Open menu by default
            self._send_slideshow_menu(chat_id)
            return
        if len(parts) >= 2 and parts[1].lower() in {"help", "?", "menu"}:
            self._send_slideshow_menu(chat_id)
            return
        action = parts[1].lower()
        if action == "start":
            # Defaults
            filt = "all"
            interval = 60
            shuffle = False
            weather = "off"
            # Parse remaining tokens
            for token in parts[2:]:
                t = token.strip()
                if t in {"all", "bg", "composite"}:
                    filt = t
                elif t.startswith("interval="):
                    try:
                        interval = max(5, int(t.split("=", 1)[1]))
                    except Exception:
                        pass
                elif t.startswith("shuffle="):
                    val = t.split("=", 1)[1].lower()
                    shuffle = val in {"on", "true", "1", "yes"}
                elif t.startswith("weather="):
                    val = t.split("=", 1)[1].lower()
                    if val in {"off", "badge", "overlay"}:
                        weather = val
            self._start_slideshow(chat_id, filt=filt, interval=interval, shuffle=shuffle, weather=weather)
        elif action == "stop":
            self._stop_slideshow(chat_id)
        else:
            self._send_message(chat_id, "Use /slideshow start … or /slideshow stop")

    def _start_slideshow(self, chat_id, filt="all", interval=60, shuffle=False, weather="off"):
        if self.slideshow_state.get("active"):
            self._send_message(chat_id, "Slideshow already running. Use /slideshow stop to end.")
            return
        # Collect saved images
        try:
            names = self.text_flow._list_saved_names()  # pylint: disable=protected-access
        except Exception:
            logger.exception("Failed to list saved images for slideshow")
            names = []
        if filt == "bg":
            names = [n for n in names if n.startswith("bg_") or n.startswith("txtbg_")]
        elif filt == "composite":
            names = [n for n in names if n.startswith("composite_")]
        if not names:
            self._send_message(chat_id, f"No saved images for filter '{filt}'.")
            return
        self.slideshow_state.update({
            "active": True,
            "filter": filt,
            "interval": interval,
            "shuffle": shuffle,
            "weather": weather,
            "was_refresh_running": bool(self.refresh_task.running),
        })
        # Pause refresh task to avoid interference
        if self.refresh_task.running:
            self.refresh_task.stop()
        # Start thread
        self.slideshow_stop.clear()
        self.slideshow_thread = threading.Thread(
            target=self._slideshow_loop,
            args=(chat_id,),
            name="TelegramSlideshow",
            daemon=True,
        )
        self.slideshow_thread.start()
        self._send_message(chat_id, f"Slideshow started: {len(names)} items | filter={filt} | interval={interval}s | shuffle={'on' if shuffle else 'off'} | weather={weather}")

    def _stop_slideshow(self, chat_id):
        if not self.slideshow_state.get("active"):
            self._send_message(chat_id, "No slideshow is running.")
            return
        self.slideshow_stop.set()
        if self.slideshow_thread and self.slideshow_thread.is_alive():
            self.slideshow_thread.join(timeout=2)
        self.slideshow_state["active"] = False
        # Resume refresh task if it was running before
        if self.slideshow_state.get("was_refresh_running") and not self.refresh_task.running:
            self.refresh_task.start()
        self._send_message(chat_id, "Slideshow stopped.")

    def _slideshow_loop(self, chat_id):
        try:
            st = dict(self.slideshow_state)
            try:
                names = self.text_flow._list_saved_names()  # pylint: disable=protected-access
            except Exception:
                logger.exception("Failed to list saved images for slideshow loop")
                names = []
            if st["filter"] == "bg":
                names = [n for n in names if n.startswith("bg_") or n.startswith("txtbg_")]
            elif st["filter"] == "composite":
                names = [n for n in names if n.startswith("composite_")]
            if not names:
                self._send_message(chat_id, "Slideshow ended: no items.")
                return
            if st["shuffle"]:
                import random
                random.shuffle(names)
            idx = 0
            while not self.slideshow_stop.is_set():
                name = names[idx % len(names)]
                idx += 1
                path = os.path.join(self.text_flow.storage_dir, "saved", f"{name}.png")
                try:
                    with Image.open(path) as img:
                        final_img = img.convert("RGB")
                    # Apply overlays per slideshow weather mode; obey composite rule
                    is_composite = name.startswith("composite_")
                    try:
                        if st["weather"] == "badge":
                            final_img = self.text_flow.overlay_weather_badge(final_img)
                        elif st["weather"] == "overlay" and not is_composite:
                            final_img = self.text_flow.overlay_weather_caption(final_img)
                        elif st["weather"] == "off":
                            # Apply global overlay if enabled and allowed
                            opts = self._get_weather_options()
                            if (opts.get("weather", {}).get("overlay", {}).get("enabled")) and not is_composite:
                                final_img = self.text_flow.overlay_weather_caption(final_img)
                    except Exception:
                        logger.exception("Slideshow overlay failed")
                    # Display
                    self.display_manager.display_image(final_img)
                    current_dt = self.refresh_task._get_current_datetime() if hasattr(self.refresh_task, "_get_current_datetime") else datetime.utcnow()
                    image_hash = compute_image_hash(final_img)
                    refresh_info = RefreshInfo(
                        refresh_type="Slideshow",
                        plugin_id="slideshow",
                        refresh_time=current_dt.isoformat(),
                        image_hash=image_hash,
                    )
                    self.device_config.refresh_info = refresh_info
                    self.device_config.write_config()
                except Exception:
                    logger.exception("Slideshow display failed for %s", name)
                # Wait interval or until stopped
                stop = self.slideshow_stop.wait(timeout=st["interval"])
                if stop:
                    break
        finally:
            self.slideshow_state["active"] = False
            # Resume refresh task if needed
            if self.slideshow_state.get("was_refresh_running") and not self.refresh_task.running:
                self.refresh_task.start()

    # --- Slideshow menu -----------------------------------------------------

    def _send_slideshow_menu(self, chat_id, message_id=None):
        st = self.slideshow_state
        lines = [
            "🖼 Slideshow",
            f"Status: {'Active' if st['active'] else 'Stopped'}",
            f"Filter: {st['filter']} | Interval: {st['interval']}s | Shuffle: {'on' if st['shuffle'] else 'off'}",
            f"Weather: {st['weather']}",
        ]
        text = "\n".join(lines)
        kb = {"inline_keyboard": [
            [
                {"text": f"Filter: {st['filter']}", "callback_data": "ss|cycle_filter"},
            ],
            [
                {"text": "-10s", "callback_data": "ss|interval|dec"},
                {"text": f"Interval: {st['interval']}s", "callback_data": "ss|noop"},
                {"text": "+10s", "callback_data": "ss|interval|inc"},
            ],
            [
                {"text": f"Shuffle: {'On' if st['shuffle'] else 'Off'}", "callback_data": "ss|toggle_shuffle"},
            ],
            [
                {"text": f"Weather: {st['weather'].title()}", "callback_data": "ss|cycle_weather"},
            ],
            [
                {"text": ("⏹ Stop" if st['active'] else "▶️ Start"), "callback_data": ("ss|stop" if st['active'] else "ss|start")},
                {"text": "✖️ Close", "callback_data": "ss|close"},
            ],
        ]}
        if message_id:
            try:
                self._api_post("editMessageText", data={
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "text": text,
                    "reply_markup": json.dumps(kb),
                })
            except Exception:
                logger.exception("Failed to edit slideshow menu; sending new one.")
                self._api_post("sendMessage", data={"chat_id": chat_id, "text": text, "reply_markup": json.dumps(kb)})
        else:
            resp = self._api_post("sendMessage", data={"chat_id": chat_id, "text": text, "reply_markup": json.dumps(kb)})
            try:
                mid = resp.get("result", {}).get("message_id")
                if mid:
                    self.ss_menu_ids[chat_id] = mid
            except Exception:
                pass

    def _refresh_slideshow_menu(self, chat_id, message_id=None):
        message_id = message_id or self.ss_menu_ids.get(chat_id)
        if message_id:
            self._send_slideshow_menu(chat_id, message_id)
        else:
            self._send_slideshow_menu(chat_id)
