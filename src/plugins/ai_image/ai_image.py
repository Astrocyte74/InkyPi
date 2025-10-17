from plugins.base_plugin.base_plugin import BasePlugin
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import requests
import logging
import os

logger = logging.getLogger(__name__)

IMAGE_MODELS = ["dall-e-3", "dall-e-2", "gpt-image-1"]
DEFAULT_IMAGE_MODEL = "dall-e-3"
DEFAULT_IMAGE_QUALITY = "standard"

SPECTRA6_INSTRUCTIONS = (
    "Generate a flat, high-contrast illustration sized 800x480 pixels using only black, white, red, green, blue, "
    "and yellow. The style must be poster-like with bold shapes, clean colour blocks, and no gradients or fine "
    "textures. Avoid subtle shading, soft edges, or photographic detail. This image will be displayed on a Spectra 6 "
    "e-ink panel with a limited colour gamut and slow refresh."
)

MONO_INSTRUCTIONS = (
    "Generate a flat, high-contrast black-and-white illustration sized 800x480 pixels. Use only pure black and pure "
    "white (no greys or gradients). Emphasise bold shapes, clear separation between light and dark areas, and avoid "
    "fine lines, soft textures, or photographic detail. The image will be displayed on a monochrome e-ink panel with "
    "a slow refresh rate."
)

VAN_GOGH_INSTRUCTIONS = (
    "Render the scene in the style of Vincent van Gogh, with expressive impasto brushstrokes, swirling motion, and "
    "vibrant post-impressionist energy."
)

ILLUSTRATION_INSTRUCTIONS = (
    "Render as a bold ink illustration with clean outlines, minimal shading, and simplified shapes suited for "
    "screen-print or poster art. Prioritise readable silhouettes and strong contrast over fine detail."
)

FAR_SIDE_INSTRUCTIONS = (
    "Illustrate as a single-panel Far Side-inspired cartoon with thick outlines, simple backgrounds, and dry humour. "
    "Use minimal text (preferably none), anthropomorphic characters, and avoid clutter or heavy shading."
)

class AIImage(BasePlugin):
    def generate_settings_template(self):
        template_params = super().generate_settings_template()
        template_params['api_key'] = {
            "required": True,
            "service": "OpenAI",
            "expected_key": "OPEN_AI_SECRET"
        }
        return template_params

    def _get_prompt_client(self, device_config, openai_client):
        open_router_key = device_config.load_env_key("OPEN_ROUTER_SECRET")
        if open_router_key:
            return {
                "type": "openrouter",
                "api_key": open_router_key,
                "model": device_config.load_env_key("OPEN_ROUTER_MODEL") or "google/gemini-2.5-flash-lite",
                "referer": device_config.load_env_key("OPEN_ROUTER_REFERRER") or "https://github.com/fatihak/InkyPi",
                "title": device_config.load_env_key("OPEN_ROUTER_TITLE") or "InkyPi"
            }
        return {"type": "openai", "client": openai_client}

    def generate_image(self, settings, device_config):

        api_key = device_config.load_env_key("OPEN_AI_SECRET")
        if not api_key:
            raise RuntimeError("OPEN AI API Key not configured.")

        text_prompt = settings.get("textPrompt", "")

        image_model = settings.get('imageModel', DEFAULT_IMAGE_MODEL)
        if image_model not in IMAGE_MODELS:
            raise RuntimeError("Invalid Image Model provided.")
        image_quality = settings.get('quality', "medium" if image_model == "gpt-image-1" else "standard")
        randomize_prompt = settings.get('randomizePrompt') == 'true'
        creative_enhance = settings.get('creativeEnhance') == 'true'
        palette = settings.get('palette', 'spectra6').lower()
        van_gogh_style = settings.get('vanGoghStyle') == 'true'
        style_hint = (settings.get('styleHint') or '').lower()
        if style_hint == 'van_gogh':
            van_gogh_style = True

        image = None
        try:
            ai_client = OpenAI(api_key = api_key)
            prompt_client = self._get_prompt_client(device_config, ai_client)
            if randomize_prompt:
                text_prompt = AIImage.fetch_image_prompt(prompt_client, text_prompt)
                if creative_enhance:
                    text_prompt = AIImage.enhance_prompt(prompt_client, text_prompt)
            elif creative_enhance:
                text_prompt = AIImage.enhance_prompt(prompt_client, text_prompt)

            if style_hint in {'van_gogh', 'illustration', 'far_side'} and not randomize_prompt:
                text_prompt = AIImage.style_polish_prompt(prompt_client, text_prompt, style_hint)

            if palette == 'bw':
                text_prompt = f"{text_prompt}. {MONO_INSTRUCTIONS}"
            else:
                # default to spectra 6 instructions
                text_prompt = f"{text_prompt}. {SPECTRA6_INSTRUCTIONS}"

            if van_gogh_style or style_hint == 'van_gogh':
                text_prompt = f"{text_prompt}. {VAN_GOGH_INSTRUCTIONS}"
            elif style_hint == 'illustration':
                text_prompt = f"{text_prompt}. {ILLUSTRATION_INSTRUCTIONS}"
            elif style_hint == 'far_side':
                text_prompt = f"{text_prompt}. {FAR_SIDE_INSTRUCTIONS}"

            image = AIImage.fetch_image(
                ai_client,
                text_prompt,
                model=image_model,
                quality=image_quality,
                orientation=device_config.get_config("orientation")
            )
        except Exception as e:
            logger.error(f"Failed to make Open AI request: {str(e)}")
            raise RuntimeError("Open AI request failure, please check logs.")
        return image

    @staticmethod
    def fetch_image(ai_client, prompt, model="dall-e-3", quality="standard", orientation="horizontal"):
        logger.info(f"Generating image for prompt: {prompt}, model: {model}, quality: {quality}")
        prompt += (
            ". The image should fully occupy the entire canvas without any frames, "
            "borders, or cropped areas. No blank spaces or artificial framing."
        )
        prompt += (
            "Focus on simplicity, bold shapes, and strong contrast to enhance clarity "
            "and visual appeal. Avoid excessive detail or complex gradients, ensuring "
            "the design works well with flat, vibrant colors."
        )
        args = {
            "model": model,
            "prompt": prompt,
            "size": "1024x1024",
        }
        if model == "dall-e-3":
            args["size"] = "1792x1024" if orientation == "horizontal" else "1024x1792"
            args["quality"] = quality
        elif model == "gpt-image-1":
            args["size"] = "1536x1024" if orientation == "horizontal" else "1024x1536"
            args["quality"] = quality

        response = ai_client.images.generate(**args)
        if model in ["dall-e-3", "dall-e-2"]:
            image_url = response.data[0].url
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
        elif model == "gpt-image-1":
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            img = Image.open(BytesIO(image_bytes))
        return img

    @staticmethod
    def fetch_image_prompt(prompt_client, from_prompt=None):
        logger.info(f"Getting random image prompt...")

        system_content = (
            "You are a creative assistant generating extremely random and unique image prompts. "
            "Avoid common themes. Focus on unexpected, unconventional, and bizarre combinations "
            "of art style, medium, subjects, time periods, and moods. No repetition. Prompts "
            "should be 20 words or less and specify random artist, movie, tv show or time period "
            "for the theme. Do not provide any headers or repeat the request, just provide the "
            "updated prompt in your response."
        )
        user_content = (
            "Give me a completely random image prompt, something unexpected and creative! "
            "Let's see what your AI mind can cook up!"
        )
        if from_prompt and from_prompt.strip():
            system_content = (
                "You are a creative assistant specializing in generating highly descriptive "
                "and unique prompts for creating images. When given a short or simple image "
                "description, your job is to rewrite it into a more detailed, imaginative, "
                "and descriptive version that captures the essence of the original while "
                "making it unique and vivid. Avoid adding irrelevant details but feel free "
                "to include creative and visual enhancements. Avoid common themes. Focus on "
                "unexpected, unconventional, and bizarre combinations of art style, medium, "
                "subjects, time periods, and moods. Do not provide any headers or repeat the "
                "request, just provide your updated prompt in the response. Prompts "
                "should be 20 words or less and specify random artist, movie, tv show or time "
                "period for the theme."
            )
            user_content = (
                f"Original prompt: \"{from_prompt}\"\n"
                "Rewrite it to make it more detailed, imaginative, and unique while staying "
                "true to the original idea. Include vivid imagery and descriptive details. "
                "Avoid changing the subject of the prompt."
            )

        prompt = AIImage._call_prompt_service(prompt_client, system_content, user_content, temperature=1)
        logger.info(f"Generated random image prompt: {prompt}")
        return prompt

    @staticmethod
    def enhance_prompt(prompt_client, prompt):
        logger.info("Enhancing image prompt for richer detail.")
        if not prompt or not prompt.strip():
            return prompt

        system_content = (
            "You rewrite image prompts to make them more descriptive and cinematic while preserving the "
            "core subject. Add lighting, composition, mood, and stylistic cues suited for poster art. "
            "Keep the result under 40 words. Do not introduce new subjects. Return only the refined prompt."
        )
        user_content = f"Refine this prompt for an illustration: \"{prompt}\""

        refined = AIImage._call_prompt_service(prompt_client, system_content, user_content, temperature=0.7)
        logger.info(f"Enhanced prompt: {refined}")
        return refined

    @staticmethod
    def style_polish_prompt(prompt_client, prompt, style_hint):
        if not prompt or not prompt.strip():
            return prompt

        style_configs = {
            "van_gogh": (
                "You are an art director converting a prompt into a vivid Vincent van Gogh style scene. Add motion, "
                "lighting, and texture cues while preserving the subject. Limit to 35 words."
            ),
            "illustration": (
                "You are an illustrator turning a prompt into a bold ink poster brief. Describe silhouettes, layout, "
                "and contrast without adding new subjects. Keep under 35 words."
            ),
            "far_side": (
                "You are a cartoonist adapting a prompt into a Far Side-inspired single-panel gag. Maintain the "
                "subject, introduce dry humour, and outline the scene in under 35 words without dialogue."
            ),
        }

        system_content = style_configs.get(
            style_hint,
            "You enhance prompts with concise art direction while preserving the subject.",
        )
        user_content = f"Original prompt: \"{prompt.strip()}\"\nRewrite it following your guidance."

        refined = AIImage._call_prompt_service(prompt_client, system_content, user_content, temperature=0.8)
        logger.info("Style polish (%s): %s", style_hint, refined)
        return refined

    @staticmethod
    def _call_prompt_service(prompt_client, system_content, user_content, temperature=0.7):
        client_type = prompt_client.get("type")
        if client_type == "openrouter":
            logger.info(
                "Prompt service: OpenRouter | model=%s | temperature=%.2f | user=%s",
                prompt_client.get("model", "google/gemini-2.5-flash-lite"),
                temperature,
                AIImage._trim_text(user_content),
            )
            return AIImage._call_openrouter(prompt_client, system_content, user_content, temperature)
        elif client_type == "openai":
            logger.info(
                "Prompt service: OpenAI gpt-4o | temperature=%.2f | user=%s",
                temperature,
                AIImage._trim_text(user_content),
            )
            return AIImage._call_openai(prompt_client["client"], system_content, user_content, temperature)
        else:
            raise RuntimeError("Unsupported prompt client configuration.")

    @staticmethod
    def _call_openai(ai_client, system_content, user_content, temperature):
        response = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _call_openrouter(prompt_client, system_content, user_content, temperature):
        headers = {
            "Authorization": f"Bearer {prompt_client['api_key']}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        referer = prompt_client.get("referer")
        title = prompt_client.get("title")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        payload = {
            "model": prompt_client.get("model", "google/gemini-2.5-flash-lite"),
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("OpenRouter request failed: %s", exc)
            raise RuntimeError("OpenRouter request failure, please check logs.") from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected OpenRouter response format: %s", data)
            raise RuntimeError("OpenRouter response parsing error, please check logs.") from exc

        if isinstance(message, list):
            message = "".join(segment.get("text", "") for segment in message)

        return str(message).strip()

    @staticmethod
    def _trim_text(text, max_length=120):
        text = text.replace("\n", " ").strip()
        if len(text) > max_length:
            return text[: max_length - 3] + "..."
        return text
