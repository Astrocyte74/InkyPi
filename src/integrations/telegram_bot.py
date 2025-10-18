import json
import os
import threading
import time
import logging
from datetime import datetime
from io import BytesIO
import shutil

import requests
from requests.exceptions import HTTPError
from PIL import Image

from model import RefreshInfo
from utils.image_utils import compute_image_hash
from plugins.plugin_registry import get_plugin_instance
from integrations.telegram_text_flow import TelegramTextFlow

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

        if text.lower() in {"/start", "/help"}:
            self._send_help(chat_id)
        elif text.lower() == "/status":
            self._send_status(chat_id)
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
                    self._api_post(
                        "sendMessage",
                        data={
                            "chat_id": request["chat_id"],
                            "text": f"Enter new name for '{name}':",
                            "reply_markup": json.dumps({"force_reply": True, "input_field_placeholder": name}),
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
        elif flow_type == "save":
            chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
            message_id = callback_query.get("message", {}).get("message_id")
            action = parts[1] if len(parts) > 1 else None
            arg = parts[2] if len(parts) > 2 else None
            if action == "choose" and arg in {"bg", "text", "txtbg"}:
                suggestion = self._suggest_save_name(arg)
                text = (
                    f"Save {'Background' if arg=='bg' else ('Composite' if arg=='text' else 'Last Background (/txt)')}\n\n"
                    f"Suggested name: {suggestion}"
                )
                markup = {
                    "inline_keyboard": [
                        [
                            {"text": f"Quick Save as '{suggestion}'", "callback_data": f"save|quick|{arg}|{suggestion}"},
                        ],
                        [
                            {"text": "📝 Enter Name…", "callback_data": f"save|prompt|{arg}"},
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
                            {"text": "Save Background", "callback_data": "save|choose|bg"},
                            {"text": "Save Text", "callback_data": "save|choose|text"},
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
        else:
            self._answer_callback(callback_query["id"])

    def _init_ai_prompt(self, chat_id, prompt, source_text_request_id=None):
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
            "source_text_request_id": source_text_request_id,
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
        saved_path = self._save_image(image)

        def send_photo(caption=None):
            try:
                with Image.open(saved_path) as img:
                    self._send_photo(request.get("chat_id"), img, caption=caption)
            except Exception:
                logger.exception("Failed to send photo back to Telegram")

        def finalize():
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
        lines = [
            "InkyPi Telegram Controls",
            "",
            "Photos:",
            "- Send a photo to update the display immediately.",
            "- /status — send the latest background image.",
            "",
            "AI Image:",
            "- /ai <prompt> — open image generator.",
            "  Configure Model, Quality, Style, Palette; then Generate.",
            "",
            "Text Composer:",
            "- /txt <message> — open text composer.",
            "  Choose style (Simple/Caption/Sticky) and Rewrite (On/Off).",
            "  Pick background:",
            "    • None — plain (centered).",
            "    • Use Last Image — use last Telegram background (text at bottom).",
            "    • Auto-Generate Image — AI background (text at bottom).",
            "    • Solid Colour — pick a colour, then 🪄 Generate.",
            "    • Saved Image — pick from saved list, 👁 Preview, 🧹 Clear, 🗑 Delete, ✏️ Rename.",
            "    • Custom Image (Prompt) — enter prompt, configure, then Generate (via /ai).",
            "",
            "Saving:",
            "- /save — interactive menu for saving images.",
            "  Options: Save Background, Save Composite, Save Last Background (/txt).",
            "  Saved files are under telegram/saved and appear in the picker.",
        ]
        self._send_message(chat_id, "\n".join(lines))

    def _send_save_menu(self, chat_id):
        text = (
            "Save Image\n\n"
            "Pick which image to save and optionally choose a name."
        )
        markup = {
            "inline_keyboard": [
                [
                    {"text": "Save Background", "callback_data": "save|choose|bg"},
                    {"text": "Save Composite", "callback_data": "save|choose|text"},
                ],
                [
                    {"text": "Save Last Background (/txt)", "callback_data": "save|choose|txtbg"},
                ],
                [
                    {"text": "✖️ Cancel", "callback_data": "save|cancel"},
                ],
            ]
        }
        self._api_post("sendMessage", data={"chat_id": chat_id, "text": text, "reply_markup": json.dumps(markup)})

    def _suggest_save_name(self, source):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        prefix = "text" if source == "text" else "bg"
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
