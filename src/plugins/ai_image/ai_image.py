from plugins.base_plugin.base_plugin import BasePlugin
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import requests
import logging

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

class AIImage(BasePlugin):
    def generate_settings_template(self):
        template_params = super().generate_settings_template()
        template_params['api_key'] = {
            "required": True,
            "service": "OpenAI",
            "expected_key": "OPEN_AI_SECRET"
        }
        return template_params

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

        image = None
        try:
            ai_client = OpenAI(api_key = api_key)
            if randomize_prompt:
                text_prompt = AIImage.fetch_image_prompt(ai_client, text_prompt)
                if creative_enhance:
                    text_prompt = AIImage.enhance_prompt(ai_client, text_prompt)
            elif creative_enhance:
                text_prompt = AIImage.enhance_prompt(ai_client, text_prompt)

            if palette == 'bw':
                text_prompt = f"{text_prompt}. {MONO_INSTRUCTIONS}"
            else:
                # default to spectra 6 instructions
                text_prompt = f"{text_prompt}. {SPECTRA6_INSTRUCTIONS}"

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
    def fetch_image_prompt(ai_client, from_prompt=None):
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

        # Make the API call
        response = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            temperature=1
        )

        prompt = response.choices[0].message.content.strip()
        logger.info(f"Generated random image prompt: {prompt}")
        return prompt

    @staticmethod
    def enhance_prompt(ai_client, prompt):
        logger.info("Enhancing image prompt for richer detail.")
        if not prompt or not prompt.strip():
            return prompt

        response = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite image prompts to make them more descriptive and cinematic while preserving the "
                        "core subject. Add lighting, composition, mood, and stylistic cues suited for poster art. "
                        "Keep the result under 40 words. Do not introduce new subjects. Return only the refined prompt."
                    )
                },
                {
                    "role": "user",
                    "content": f"Refine this prompt for an illustration: \"{prompt}\""
                }
            ],
            temperature=0.7
        )
        refined = response.choices[0].message.content.strip()
        logger.info(f"Enhanced prompt: {refined}")
        return refined
