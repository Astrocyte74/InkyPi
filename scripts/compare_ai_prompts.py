#!/usr/bin/env python3

"""Utility to compare AI image prompt rewrites between OpenRouter and OpenAI."""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for root in (REPO_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from config import Config  # noqa: E402  pylint: disable=wrong-import-position
from plugins.ai_image.ai_image import (  # noqa: E402  pylint: disable=wrong-import-position
    AIImage,
    DRAWING_INSTRUCTIONS,
    FAR_SIDE_INSTRUCTIONS,
    ILLUSTRATION_INSTRUCTIONS,
    MONO_INSTRUCTIONS,
    SPECTRA6_INSTRUCTIONS,
    VAN_GOGH_INSTRUCTIONS,
)
from plugins.plugin_registry import load_plugins, get_plugin_instance  # noqa: E402  pylint: disable=wrong-import-position


FILL_CANVAS_SENTENCE = (
    "The image should fully occupy the entire canvas without any frames, borders, or cropped areas."
)
SIMPLICITY_SENTENCE = (
    "Focus on simplicity, bold shapes, and strong contrast. Avoid excessive detail or complex gradients."
)

STYLE_LABELS = {
    None: "None",
    "randomize": "Randomize",
    "creative": "Creative",
    "van_gogh": "Van Gogh",
    "illustration": "Illustration",
    "drawing": "Drawing",
    "far_side": "Far Side",
}


@dataclass(frozen=True)
class PromptVariant:
    key: str
    label: str
    randomize: bool = False
    creative: bool = False
    style: Optional[str] = None


ALL_PROMPT_VARIANTS: Sequence[PromptVariant] = (
    PromptVariant("none", STYLE_LABELS[None]),
    PromptVariant("randomize", STYLE_LABELS["randomize"], randomize=True),
    PromptVariant("creative", STYLE_LABELS["creative"], creative=True),
    PromptVariant("van_gogh", STYLE_LABELS["van_gogh"], style="van_gogh"),
    PromptVariant("illustration", STYLE_LABELS["illustration"], style="illustration"),
    PromptVariant("drawing", STYLE_LABELS["drawing"], style="drawing"),
    PromptVariant("far_side", STYLE_LABELS["far_side"], style="far_side"),
)


def ensure_openai_override():
    if getattr(AIImage, "_openai_override_installed", False):
        return

    def call_openai_override(ai_client, system_content, user_content, temperature):
        model_name = getattr(AIImage, "_prompt_model_override", None) or "gpt-4o"
        response = ai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    AIImage._call_openai = staticmethod(call_openai_override)  # type: ignore[attr-defined]
    AIImage._openai_override_installed = True  # type: ignore[attr-defined]


def ensure_env():
    """Load environment variables from expected locations."""
    custom_path = os.environ.get("INKYPI_DOTENV_PATH")
    if custom_path:
        load_dotenv(custom_path, override=True)
    pi_path = Path("/usr/local/inkypi/.env")
    if pi_path.exists():
        load_dotenv(pi_path, override=True)
    load_dotenv(override=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare AI image prompt rewrites across providers.")
    parser.add_argument("prompt", nargs="?", help="Base text prompt")
    parser.add_argument(
        "--palette",
        choices=["colour", "color", "bw"],
        default="colour",
        help="Palette guidance to simulate (default: colour)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--randomize", action="store_true", help="Use randomize mode (fetch a fresh prompt).")
    group.add_argument("--creative", action="store_true", help="Use creative enhance mode.")
    group.add_argument(
        "--style",
        choices=["van_gogh", "illustration", "drawing", "far_side"],
        help="Apply a specific style rewrite.",
    )
    parser.add_argument(
        "--orientation",
        choices=["horizontal", "vertical"],
        help="Override device orientation for final prompt size hint.",
    )
    parser.add_argument(
        "--all-styles",
        action="store_true",
        help="Run the prompt against every available style option (overrides --randomize/--creative/--style).",
    )
    parser.add_argument(
        "--openrouter-model",
        action="append",
        dest="openrouter_models",
        help="Additional OpenRouter prompt models to include (repeatable).",
    )
    parser.add_argument(
        "--openai-model",
        action="append",
        dest="openai_models",
        help="Additional OpenAI prompt chat models to include (repeatable).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch an interactive wizard to choose prompt, palette, styles, and models.",
    )
    return parser.parse_args()


def resolve_variants(args) -> Sequence[PromptVariant]:
    style_keys = getattr(args, "style_keys", None)
    if style_keys:
        return tuple(variant for variant in ALL_PROMPT_VARIANTS if variant.key in style_keys)
    if args.all_styles:
        return ALL_PROMPT_VARIANTS

    randomize = args.randomize
    creative = args.creative
    style = args.style

    if randomize:
        return (PromptVariant("randomize", STYLE_LABELS["randomize"], randomize=True),)
    if creative:
        return (PromptVariant("creative", STYLE_LABELS["creative"], creative=True),)
    if style:
        return (PromptVariant(style, STYLE_LABELS[style], style=style),)
    return (PromptVariant("none", STYLE_LABELS[None]),)


def build_prompt(
    prompt_client,
    variant: PromptVariant,
    display_guidance: bool,
    palette_key: str,
    base_prompt: str,
) -> dict:
    """Replicate AIImage prompt composition without triggering image generation."""
    previous_override = getattr(AIImage, "_prompt_model_override", None)
    if prompt_client.get("type") == "openai":
        ensure_openai_override()
        AIImage._prompt_model_override = prompt_client.get("model")  # type: ignore[attr-defined]
    else:
        AIImage._prompt_model_override = None  # type: ignore[attr-defined]

    text_prompt = base_prompt

    if variant.randomize:
        text_prompt = AIImage.fetch_image_prompt(prompt_client, text_prompt)
        transform_label = "Randomized prompt"
    elif variant.creative:
        text_prompt = AIImage.enhance_prompt(prompt_client, text_prompt)
        transform_label = "Creative enhance prompt"
    elif variant.style:
        text_prompt = AIImage.style_rewrite_prompt(prompt_client, text_prompt, variant.style)
        transform_label = f"Style rewrite ({STYLE_LABELS.get(variant.style, variant.style)})"
    else:
        transform_label = "Prompt (unchanged)"

    final_prompt = text_prompt
    if display_guidance:
        if palette_key == "bw":
            final_prompt = f"{final_prompt}. {MONO_INSTRUCTIONS}"
        else:
            final_prompt = f"{final_prompt}. {SPECTRA6_INSTRUCTIONS}"

    style_hint = variant.style or ""
    if style_hint == "van_gogh":
        final_prompt = f"{final_prompt}. {VAN_GOGH_INSTRUCTIONS}"
    elif style_hint == "illustration":
        final_prompt = f"{final_prompt}. {ILLUSTRATION_INSTRUCTIONS}"
    elif style_hint == "drawing":
        final_prompt = f"{final_prompt}. {DRAWING_INSTRUCTIONS}"
    elif style_hint == "far_side":
        final_prompt = f"{final_prompt}. {FAR_SIDE_INSTRUCTIONS}"

    image_prompt = final_prompt
    if display_guidance:
        image_prompt = (
            f"{image_prompt}. {FILL_CANVAS_SENTENCE} "
            f"{SIMPLICITY_SENTENCE}"
        )

    result = {
        "transform_label": transform_label,
        "transformed_prompt": text_prompt,
        "image_prompt": image_prompt,
    }

    AIImage._prompt_model_override = previous_override  # type: ignore[attr-defined]
    return result


def pretty_print_result(variant_label: str, provider_label: str, result: dict):
    header = f"{variant_label} | {provider_label}".strip()
    print(f"\n=== {header} ===")
    print(f"{result['transform_label']}:")
    print(result["transformed_prompt"])
    print("\nFinal image prompt:")
    print(result["image_prompt"])


def format_prompt_clients(
    plugin,
    device_config,
    client: OpenAI,
    openrouter_models: Optional[Sequence[str]] = None,
    openai_models: Optional[Sequence[str]] = None,
) -> Sequence[Tuple[str, str, dict]]:
    prompt_client_router = plugin._get_prompt_client(device_config, client)  # pylint: disable=protected-access
    prompt_clients = []
    if prompt_client_router.get("type") == "openrouter":
        base_router = dict(prompt_client_router)
        if openrouter_models:
            for model_name in openrouter_models:
                variant = dict(base_router)
                variant["model"] = model_name
                prompt_clients.append(("OpenRouter", f"Model: {model_name}", variant))
        else:
            model_name = base_router.get("model", "unknown")
            prompt_clients.append(("OpenRouter", f"Model: {model_name}", base_router))
    else:
        if openrouter_models:
            raise SystemExit("--openrouter-model specified but OpenRouter is not configured in the environment.")
        prompt_clients.append(("OpenRouter (fallback to OpenAI)", "", {"type": "openai", "client": client}))

    base_openai_models = ["gpt-4o"]
    if openai_models:
        base_openai_models = list(dict.fromkeys(base_openai_models + list(openai_models)))

    for model_name in base_openai_models:
        prompt_clients.append(
            (
                "OpenAI",
                f"Model: {model_name}",
                {"type": "openai", "client": client, "model": model_name},
            )
        )
    return prompt_clients


def _input_with_default(prompt_text: str, default: Optional[str] = None) -> str:
    message = prompt_text
    if default:
        message += f" [{default}]"
    message += ": "
    response = input(message).strip()
    return response or (default or "")


def _interactive_select_styles() -> Sequence[str]:
    print("\nAvailable styles:")
    for idx, variant in enumerate(ALL_PROMPT_VARIANTS, start=1):
        print(f"  {idx}. {variant.label}")
    print("  a. All styles")
    selection = input("Select styles (comma separated, Enter for 'None' only): ").strip().lower()
    if not selection:
        return ["none"]
    if selection in {"a", "all"}:
        return [variant.key for variant in ALL_PROMPT_VARIANTS]

    chosen = []
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            print(f"Ignoring invalid selection: {part}")
            continue
        index = int(part)
        if 1 <= index <= len(ALL_PROMPT_VARIANTS):
            chosen.append(ALL_PROMPT_VARIANTS[index - 1].key)
        else:
            print(f"Ignoring out-of-range selection: {part}")
    return chosen or ["none"]


def interactive_config(
    args,
    plugin,
    device_config,
    client: OpenAI,
) -> None:
    print("Interactive prompt comparison wizard")
    print("------------------------------------")

    if not args.prompt:
        while True:
            prompt = input("Enter base prompt: ").strip()
            if prompt:
                args.prompt = prompt
                break
            print("Prompt cannot be empty.")

    args.palette = _input_with_default("Palette (colour/bw)", default=args.palette or "colour").lower()
    if args.palette not in {"colour", "color", "bw"}:
        print("Unknown palette, defaulting to 'colour'.")
        args.palette = "colour"

    default_orientation = device_config.get_config("orientation") or "horizontal"
    orientation_choice = _input_with_default("Orientation (horizontal/vertical)", default=default_orientation).lower()
    if orientation_choice not in {"horizontal", "vertical"}:
        print("Unknown orientation, defaulting to device setting.")
        orientation_choice = default_orientation
    args.orientation = orientation_choice

    style_keys = _interactive_select_styles()
    if style_keys:
        if len(style_keys) == len(ALL_PROMPT_VARIANTS):
            args.all_styles = True
        else:
            args.style_keys = style_keys
            args.all_styles = False
            args.randomize = False
            args.creative = False
            args.style = None

    router_client = plugin._get_prompt_client(device_config, client)  # pylint: disable=protected-access
    if router_client.get("type") == "openrouter":
        current_model = router_client.get("model")
        print(f"\nConfigured OpenRouter model: {current_model or '(not set)'}")
        print("Enter comma-separated OpenRouter models to compare or press Enter to use the configured default.")
        router_input = input("OpenRouter models: ").strip()
        if router_input:
            args.openrouter_models = [model.strip() for model in router_input.split(",") if model.strip()]
    else:
        print("\nOpenRouter is not configured; skipping OpenRouter model selection.")

    print("\nDefault OpenAI prompt model: gpt-4o")
    openai_input = input("Enter comma-separated additional OpenAI models (Enter for none): ").strip()
    if openai_input:
        args.openai_models = [model.strip() for model in openai_input.split(",") if model.strip()]


def main():
    ensure_env()
    args = parse_args()
    device_config = Config()
    load_plugins(device_config.get_plugins())
    plugin_config = device_config.get_plugin("ai_image")
    if not plugin_config:
        raise SystemExit("AI Image plugin is not installed.")

    plugin = get_plugin_instance(plugin_config)

    openai_key = device_config.load_env_key("OPEN_AI_SECRET")
    if not openai_key:
        raise SystemExit("OPEN_AI_SECRET is not configured.")

    client = OpenAI(api_key=openai_key)

    if args.interactive:
        interactive_config(args, plugin, device_config, client)

    if not args.prompt:
        raise SystemExit("Prompt is required.")

    if args.all_styles and (args.randomize or args.creative or args.style) and not getattr(args, "style_keys", None):
        raise SystemExit("--all-styles cannot be combined with --randomize/--creative/--style.")

    orientation = args.orientation or device_config.get_config("orientation") or "horizontal"
    palette_key = "spectra6" if args.palette in {"colour", "color"} else "bw"
    display_guidance = AIImage._display_guidance_enabled(device_config)  # pylint: disable=protected-access

    variants = resolve_variants(args)
    prompt_clients = format_prompt_clients(
        plugin,
        device_config,
        client,
        openrouter_models=args.openrouter_models,
        openai_models=args.openai_models,
    )

    print("Base prompt:")
    print(args.prompt)
    print(f"\nPalette: {'Colour' if palette_key == 'spectra6' else 'Black & White'}")
    print(f"Orientation: {orientation}")
    print(f"Display guidance enabled: {display_guidance}")
    if len(variants) > 1:
        print(f"Running {len(variants)} styles.\n")

    for variant in variants:
        for name, detail, prompt_client in prompt_clients:
            label = f"{name} {detail}".strip()
            try:
                result = build_prompt(
                    prompt_client,
                    variant,
                    display_guidance,
                    palette_key,
                    args.prompt,
                )
                pretty_print_result(variant.label, label, result)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"\n=== {variant.label} | {label} ===")
                print(f"Failed: {exc}")


if __name__ == "__main__":
    main()
