# hf-shared-utils

Shared utilities for Hugging Face Spaces and Gradio apps, focused on LLM model/provider configuration UI and OpenAI-compatible client setup.

**Install**
```sh
pip install -e .
```

**Dev Setup**
- Python `>=3.10` (pre-commit is configured to use `python3.12` by default).
- Install dev tooling:
```sh
pip install -e ".[dev]"
```
- Enable git hooks:
```sh
pre-commit install
```

**Run Tests**
```sh
pytest
```

**Run Pre-commit Manually**
```sh
pre-commit run --all-files
```

**What’s Inside**
`hf_shared_utils.llm_model_config` provides:
- A drop-in Gradio UI section for provider, model, and API key inputs.
- Helpers to resolve model names and build an OpenAI-compatible client with a provider-specific base URL.

**Quick Start**
```python
import gradio as gr
from hf_shared_utils.llm_model_config import (
    build_model_config_section,
    build_model_config_updates,
    create_openai_client,
    resolve_model_name,
)


def run(prompt, provider, provider_url, model_choice, model_name_custom, api_key):
    model = resolve_model_name(model_choice, model_name_custom)
    client = create_openai_client(provider, provider_url, api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Prompt")
    provider, provider_url, model_choice, model_name_custom, api_key = build_model_config_section(
        default_provider="openai",
        default_model="gpt-4o-mini",
        default_provider_url="",
    )
    output = gr.Textbox(label="Output")

    gr.Button("Run").click(
        run,
        inputs=[prompt, provider, provider_url, model_choice, model_name_custom, api_key],
        outputs=output,
    )

demo.launch()
```

**Providers And Defaults**
The provider dropdown and base URLs are defined in `hf_shared_utils/llm_model_config/llm_model_config.py`.
Current providers:
- `openai`
- `anthropic`
- `google`
- `deepseek`
- `mistral`
- `cohere`
- `groq`
- `grok`
- `openrouter`
- `minimax`
- `custom`

Each provider has a small list of popular models used to prefill the model dropdown, plus a `custom` option.

**API**
- `build_model_config_section(default_provider=None, default_model=None, default_provider_url=None)`
  Returns the Gradio components: `provider`, `provider_custom_url`, `model_choice`, `model_name_custom`, `api_key`.
  `default_provider` and `default_model` let the consumer preselect values. If `default_model` is not in the provider's list, the UI switches to `custom` and fills the custom model field.
  `default_provider_url` pre-fills the Provider URL textbox (shown when provider is `custom`).
- `build_model_config_updates(provider=None, model_name=None, provider_url=None, current_provider=None)`
  Returns `gr.update(...)` objects for `provider`, `provider_custom_url`, `model_choice`, and `model_name_custom`.
  Use this to change values dynamically after render. All inputs are optional; any omitted fields are left unchanged.
- `resolve_model_name(model_choice, model_name_custom)`
  Returns the selected model string, falling back to the custom input when `model_choice == "custom"`.
- `create_openai_client(provider, custom_base_url, api_key, verify_ssl=True)`
  Returns an `OpenAI` client configured with the provider base URL. Raises `ValueError` if the API key is missing, or if `provider == "custom"` and the base URL is empty.
  Set `verify_ssl=False` to disable SSL certificate verification (useful for self-signed certs or local endpoints).

**Dynamic Updates Example**
```python
def apply_defaults(provider_override, model_override, current_provider):
    return build_model_config_updates(
        provider=provider_override,
        model_name=model_override,
        current_provider=current_provider,
    )

with gr.Blocks() as demo:
    provider, provider_url, model_choice, model_name_custom, api_key = build_model_config_section()
    provider_override = gr.Textbox(label="Provider Override")
    model_override = gr.Textbox(label="Model Override")

    gr.Button("Apply").click(
        apply_defaults,
        inputs=[provider_override, model_override, provider],
        outputs=[provider, provider_url, model_choice, model_name_custom],
    )
```

**Notes**
`build_model_config_section()` also renders a short security notice aimed at Hugging Face Spaces. Update or remove it if your app has different requirements.

**Requirements**
- Python `>=3.10`
- `gradio>=4`
- `openai>=1`
