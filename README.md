# hf-shared-utils

Shared utilities for Hugging Face Spaces and Gradio apps, focused on LLM model/provider configuration UI and OpenAI-compatible client setup.

**Install**
```sh
pip install -e .
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
    provider, provider_url, model_choice, model_name_custom, api_key = build_model_config_section()
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
- `groq`
- `grok`
- `openrouter`
- `custom`

Each provider has a small list of popular models used to prefill the model dropdown, plus a `custom` option.

**API**
- `build_model_config_section()`
  Returns the Gradio components: `provider`, `provider_custom_url`, `model_choice`, `model_name_custom`, `api_key`.
- `resolve_model_name(model_choice, model_name_custom)`
  Returns the selected model string, falling back to the custom input when `model_choice == "custom"`.
- `create_openai_client(provider, custom_base_url, api_key)`
  Returns an `OpenAI` client configured with the provider base URL. Raises `ValueError` if the API key is missing, or if `provider == "custom"` and the base URL is empty.

**Notes**
`build_model_config_section()` also renders a short security notice aimed at Hugging Face Spaces. Update or remove it if your app has different requirements.

**Requirements**
- Python `>=3.10`
- `gradio>=4`
- `openai>=1`
