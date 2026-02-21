from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


# Ensure repo root is on sys.path so tests can import the local package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_gradio_stub():
    gr = ModuleType("gradio")

    class _DummyContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _update(**kwargs):
        return dict(kwargs)

    def _noop(*args, **kwargs):
        return object()

    gr.Markdown = _noop
    gr.update = _update
    gr.Row = _DummyContext
    gr.Dropdown = _noop
    gr.Textbox = _noop

    sys.modules["gradio"] = gr


def _install_openai_stub():
    openai = ModuleType("openai")

    class OpenAI:  # minimal stub for import-time usage
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key
            self.base_url = base_url

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


# Provide stubs so tests can run without heavy runtime deps installed.
try:
    import gradio  # noqa: F401
except Exception:
    _install_gradio_stub()

try:
    import openai  # noqa: F401
except Exception:
    _install_openai_stub()
