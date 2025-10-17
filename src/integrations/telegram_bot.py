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

logger = logging.getLogger(__name__)


class TelegramBotListener:
    """Background poller that listens for Telegram messages and triggers display updates."""

    API_BASE = "https://api.telegram.org/bot{token}"
    FILE_BASE = "https://api.telegram.org/file/bot{token}"

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
                params = {"timeout": self.poll_timeout, "allowed_updates": ["message", "edited_message"]}
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
        if not message:
            return

        chat_id = message["chat"]["id"]
        user_id = message.get("from", {}).get("id")

        if self.allowed_ids and user_id not in self.allowed_ids:
            logger.warning("Ignoring message from unauthorized user_id=%s", user_id)
            self._send_message(chat_id, "Unauthorized user.")
            return

        if "text" in message:
            self._handle_text(message["text"], chat_id)
        elif "photo" in message:
            self._handle_photo(message["photo"], chat_id)
        else:
            self._send_message(chat_id, "Send a photo or use /status.")

    def _handle_text(self, text, chat_id):
        text = text.strip()
        if text.lower() in {"/start", "/help"}:
            self._send_message(
                chat_id,
                "Send a photo to update the InkyPi display, or use /status to fetch the latest rendered image.",
            )
        elif text.lower() == "/status":
            self._send_status(chat_id)
        else:
            self._send_message(chat_id, "Unknown command. Send a photo or use /status.")

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
        self._api_post(
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

