from typing import Dict, List, Optional, Tuple
import gradio as gr
from openai import OpenAI


# ----------------------------
# Security Notice (internal)
# ----------------------------
def build_security_notice():
    SECURITY_NOTICE_TEXT = (
        "This demo runs on Hugging Face Spaces.<br>"
        "Your API key is used only to run this session and is not stored.<br>"
        "All source code is public and can be inspected or self-hosted."
    )
    notice_markup = (
        "<div style='"
        "margin-top: 0.75rem;"
        "color: #8f8f94;"
        "font-size: 0.84rem;"
        "line-height: 1.45;"
        "'>"
        "<div style='margin-bottom: 0.25rem; font-weight: 600; color: #8f8f94;'>Disclaimer</div>"
        f"{SECURITY_NOTICE_TEXT}"
        "</div>"
    )

    # Use borderless markdown container when available.
    # Some Gradio versions do not accept `container`.
    try:
        return gr.Markdown(notice_markup, container=False)
    except TypeError:
        return gr.Markdown(notice_markup)


# ----------------------------
# Providers
# ----------------------------
BASE_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1/",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "deepseek": "https://api.deepseek.com",
    "groq": "https://api.groq.com/openai/v1",
    "grok": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": "",
}

PROVIDERS: List[str] = [
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "groq",
    "grok",
    "openrouter",
    "custom",
]

POPULAR_MODELS: Dict[str, List[str]] = {
    # OpenAI model IDs (see Models page)
    # https://developers.openai.com/api/docs/models
    "openai": [
        # GPT-5 family (latest recommended)
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        # GPT-4.1 family
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        # GPT-4o family
        "gpt-4o",
        "gpt-4o-mini",
        # Reasoning models
        "o3",
        "o4-mini",
    ],

    # Anthropic Claude API IDs / aliases (see Models overview)
    # https://platform.claude.com/docs/en/about-claude/models/overview
    "anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        # Snapshot IDs (useful when you want deterministic behavior)
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    ],

    # Google Gemini model codes (OpenAI-compat endpoint supported)
    # https://ai.google.dev/gemini-api/docs/models
    "google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        # Useful previews / special modalities
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash-lite-preview-09-2025",
        "gemini-2.5-pro-preview-tts",
        "gemini-2.5-flash-preview-tts",
        "gemini-2.5-flash-image",
        "gemini-2.5-flash-native-audio-preview-12-2025",
    ],

    # DeepSeek model IDs (see API docs)
    # https://api-docs.deepseek.com/
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],

    # Groq (OpenAI-compatible) model IDs (see Groq docs)
    # https://console.groq.com/docs/models
    "groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ],

    # xAI Grok model IDs (see xAI docs)
    # https://docs.x.ai/developers/models
    "grok": [
        "grok-4",
        "grok-4-latest",
        "grok-3",
        "grok-3-mini",
    ],

    # OpenRouter uses provider-prefixed model IDs.
    # Keep this list short (OpenRouter has many models and changes frequently).
    "openrouter": [
        "openai/gpt-5.2",
        "openai/gpt-4.1",
        "anthropic/claude-opus-4-6",
        "google/gemini-2.5-pro",
        "deepseek/deepseek-reasoner",
        "x-ai/grok-4",
    ],

    "custom": [],
}


# ----------------------------
# Resolution helpers
# ----------------------------
def resolve_base_url(provider: str, custom_base_url: str) -> str:
    p = (provider or "").strip().lower()
    if p == "custom":
        return (custom_base_url or "").strip()
    return BASE_URLS.get(p, "")


def resolve_model_name(model_choice: str, model_name_custom: str) -> str:
    if model_choice == "custom":
        return (model_name_custom or "").strip()
    return (model_choice or "").strip()


def resolve_model_defaults(
    provider: Optional[str],
    model_name: Optional[str],
) -> Tuple[str, List[str], str, str]:
    provider_value = (provider or "").strip().lower()
    if not provider_value:
        provider_value = "openai"
    if provider_value not in PROVIDERS:
        provider_value = "custom"

    model_value = (model_name or "").strip()

    base_models = POPULAR_MODELS.get(provider_value, [])
    model_choices = base_models + ["custom"] if base_models else ["custom"]

    if model_value:
        if model_value in base_models:
            return provider_value, model_choices, model_value, ""
        return provider_value, model_choices, "custom", model_value

    if base_models:
        return provider_value, model_choices, base_models[0], ""

    return provider_value, model_choices, "custom", ""


def _normalize_optional(value: Optional[str]) -> Optional[str]:
    normalized = (value or "").strip()
    return normalized or None


def create_openai_client(provider: str, custom_base_url: str, api_key: str):
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("Missing API key.")

    base_url = resolve_base_url(provider, custom_base_url)

    if (provider or "").strip().lower() == "custom" and not base_url:
        raise ValueError("Custom provider requires Base URL.")

    return OpenAI(api_key=api_key, base_url=base_url or None)


# ----------------------------
# UI Helpers
# ----------------------------
def models_for_provider(provider: str):
    p = (provider or "").strip().lower()
    base = POPULAR_MODELS.get(p, [])
    choices = base + ["custom"]
    value = base[0] if base else "custom"
    return gr.update(choices=choices, value=value)


def show_custom_fields(provider: str, model_choice: str):
    provider_is_custom = ((provider or "").strip().lower() == "custom")
    model_is_custom = (model_choice == "custom")
    return (
        gr.update(visible=provider_is_custom),
        gr.update(visible=model_is_custom),
    )


def build_model_config_updates(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    provider_url: Optional[str] = None,
    current_provider: Optional[str] = None,
):
    provider_override = _normalize_optional(provider)
    model_override = _normalize_optional(model_name)
    current_provider_value = _normalize_optional(current_provider)

    provider_for_models = provider_override or current_provider_value

    provider_update = gr.update()
    provider_url_kwargs: Dict[str, object] = {}
    model_choice_kwargs: Dict[str, object] = {}
    model_custom_kwargs: Dict[str, object] = {}

    if provider_override is not None or model_override is not None:
        if provider_for_models:
            provider_value, model_choices, model_choice_value, model_custom_value = resolve_model_defaults(
                provider_for_models,
                model_override,
            )
        else:
            provider_value = None
            model_choices = None
            model_choice_value = "custom" if model_override else None
            model_custom_value = model_override or ""

        if provider_override is not None and provider_value:
            provider_update = gr.update(value=provider_value)
            provider_url_kwargs["visible"] = provider_value == "custom"
            if model_choices is not None:
                model_choice_kwargs["choices"] = model_choices

        if model_choice_value is not None:
            model_choice_kwargs["value"] = model_choice_value
            model_custom_kwargs["visible"] = model_choice_value == "custom"
        if model_custom_value is not None:
            model_custom_kwargs["value"] = model_custom_value

    if provider_url is not None:
        provider_url_kwargs["value"] = (provider_url or "").strip()
        if "visible" not in provider_url_kwargs and current_provider_value is not None:
            provider_url_kwargs["visible"] = current_provider_value == "custom"

    provider_url_update = gr.update(**provider_url_kwargs) if provider_url_kwargs else gr.update()
    model_choice_update = gr.update(**model_choice_kwargs) if model_choice_kwargs else gr.update()
    model_custom_update = gr.update(**model_custom_kwargs) if model_custom_kwargs else gr.update()

    return provider_update, provider_url_update, model_choice_update, model_custom_update


def build_model_config_row(
    default_provider: Optional[str] = None,
    default_model: Optional[str] = None,
    default_provider_url: Optional[str] = None,
):
    provider_value, model_choices, model_choice_value, model_name_custom_value = resolve_model_defaults(
        default_provider,
        default_model,
    )
    provider_is_custom = provider_value == "custom"
    model_is_custom = model_choice_value == "custom"

    with gr.Row():
        provider = gr.Dropdown(
            label="Provider",
            choices=PROVIDERS,
            value=provider_value,
            scale=2,
        )

        provider_custom_url = gr.Textbox(
            label="Provider URL",
            placeholder="https://api.openai.com/v1",
            value=(default_provider_url or "").strip(),
            visible=provider_is_custom,
            scale=4,
        )

        model_choice = gr.Dropdown(
            label="Model",
            choices=model_choices,
            value=model_choice_value,
            scale=3,
        )

        model_name_custom = gr.Textbox(
            label="Model Name",
            placeholder="e.g. gpt-4o-mini",
            value=model_name_custom_value,
            visible=model_is_custom,
            scale=3,
        )

        api_key = gr.Textbox(
            label="API Key",
            type="password",
            placeholder="Paste your key",
            scale=4,
        )

    return provider, provider_custom_url, model_choice, model_name_custom, api_key


def wire_model_config_events(provider, provider_custom_url, model_choice, model_name_custom):
    provider.change(models_for_provider, inputs=provider, outputs=model_choice).then(
        show_custom_fields,
        inputs=[provider, model_choice],
        outputs=[provider_custom_url, model_name_custom],
    )

    model_choice.change(
        show_custom_fields,
        inputs=[provider, model_choice],
        outputs=[provider_custom_url, model_name_custom],
    )


# ----------------------------
# Public Section Builder (ONLY thing app.py should call)
# ----------------------------
def build_model_config_section(
    default_provider: Optional[str] = None,
    default_model: Optional[str] = None,
    default_provider_url: Optional[str] = None,
):
    with gr.Group():
        provider, provider_custom_url, model_choice, model_name_custom, api_key = build_model_config_row(
            default_provider=default_provider,
            default_model=default_model,
            default_provider_url=default_provider_url,
        )
        build_security_notice()

    wire_model_config_events(provider, provider_custom_url, model_choice, model_name_custom)

    return provider, provider_custom_url, model_choice, model_name_custom, api_key
