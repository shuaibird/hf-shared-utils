import pytest

from hf_shared_utils.llm_model_config import llm_model_config as mod


def _get_update_attr(update, name):
    if isinstance(update, dict):
        return update.get(name)
    return getattr(update, name)


def test_resolve_base_url_known_provider():
    assert mod.resolve_base_url("OpenAI", "") == mod.BASE_URLS["openai"]


def test_resolve_base_url_custom_trimmed():
    assert mod.resolve_base_url("custom", "  https://example.com/v1  ") == "https://example.com/v1"


def test_resolve_base_url_unknown_provider():
    assert mod.resolve_base_url("unknown", "https://example.com") == ""


def test_resolve_model_name_custom_trimmed():
    assert mod.resolve_model_name("custom", "  gpt-test  ") == "gpt-test"


def test_resolve_model_name_non_custom_trimmed():
    assert mod.resolve_model_name("  gpt-test  ", "ignored") == "gpt-test"


def test_create_openai_client_missing_api_key():
    with pytest.raises(ValueError, match="Missing API key."):
        mod.create_openai_client("openai", "", "")


def test_create_openai_client_custom_missing_base_url():
    with pytest.raises(ValueError, match="Custom provider requires Base URL."):
        mod.create_openai_client("custom", "", "sk-test")


def test_create_openai_client_sets_base_url_for_known_provider(monkeypatch):
    class DummyClient:
        def __init__(self, api_key, base_url=None, http_client=None):
            self.api_key = api_key
            self.base_url = base_url

    monkeypatch.setattr(mod, "OpenAI", DummyClient)

    client = mod.create_openai_client("openai", "", "  sk-test  ")

    assert isinstance(client, DummyClient)
    assert client.api_key == "sk-test"
    assert client.base_url == mod.BASE_URLS["openai"]


def test_create_openai_client_custom_base_url(monkeypatch):
    class DummyClient:
        def __init__(self, api_key, base_url=None, http_client=None):
            self.api_key = api_key
            self.base_url = base_url

    monkeypatch.setattr(mod, "OpenAI", DummyClient)

    client = mod.create_openai_client("custom", "  https://custom.local/v1  ", "sk-test")

    assert client.base_url == "https://custom.local/v1"


def test_models_for_provider_known():
    update = mod.models_for_provider("openai")
    choices = _get_update_attr(update, "choices")
    value = _get_update_attr(update, "value")

    assert choices[-1] == "custom"
    assert value == mod.POPULAR_MODELS["openai"][0]


def test_models_for_provider_unknown():
    update = mod.models_for_provider("unknown")
    choices = _get_update_attr(update, "choices")
    value = _get_update_attr(update, "value")

    assert choices == ["custom"]
    assert value == "custom"


def test_show_custom_fields_visibility():
    provider_update, model_update = mod.show_custom_fields("custom", "custom")

    assert _get_update_attr(provider_update, "visible") is True
    assert _get_update_attr(model_update, "visible") is True


def test_show_custom_fields_non_custom():
    provider_update, model_update = mod.show_custom_fields("openai", "gpt-4o-mini")

    assert _get_update_attr(provider_update, "visible") is False
    assert _get_update_attr(model_update, "visible") is False


def test_resolve_model_defaults_known_provider_known_model():
    provider, choices, model_choice, model_custom = mod.resolve_model_defaults("openai", "gpt-4o")

    assert provider == "openai"
    assert "gpt-4o" in choices
    assert choices[-1] == "custom"
    assert model_choice == "gpt-4o"
    assert model_custom == ""


def test_resolve_model_defaults_known_provider_custom_model():
    provider, choices, model_choice, model_custom = mod.resolve_model_defaults("openai", "my-model")

    assert provider == "openai"
    assert "custom" in choices
    assert model_choice == "custom"
    assert model_custom == "my-model"


def test_resolve_model_defaults_unknown_provider():
    provider, choices, model_choice, model_custom = mod.resolve_model_defaults("unknown", "my-model")

    assert provider == "custom"
    assert choices == ["custom"]
    assert model_choice == "custom"
    assert model_custom == "my-model"


def test_resolve_model_defaults_empty_inputs():
    provider, choices, model_choice, model_custom = mod.resolve_model_defaults("", "")

    assert provider == "openai"
    assert choices[-1] == "custom"
    assert model_choice == mod.POPULAR_MODELS["openai"][0]
    assert model_custom == ""


def test_build_model_config_updates_sets_provider_and_model():
    provider_update, provider_url_update, model_choice_update, model_custom_update = (
        mod.build_model_config_updates(provider="openai", model_name="gpt-4o")
    )

    assert _get_update_attr(provider_update, "value") == "openai"
    assert _get_update_attr(model_choice_update, "value") == "gpt-4o"
    assert "gpt-4o" in _get_update_attr(model_choice_update, "choices")
    assert _get_update_attr(model_custom_update, "visible") is False
    assert _get_update_attr(model_custom_update, "value") == ""


def test_build_model_config_updates_custom_model_with_current_provider():
    provider_update, provider_url_update, model_choice_update, model_custom_update = (
        mod.build_model_config_updates(model_name="my-model", current_provider="openai")
    )

    assert _get_update_attr(provider_update, "value") is None
    assert _get_update_attr(model_choice_update, "value") == "custom"
    assert _get_update_attr(model_custom_update, "visible") is True
    assert _get_update_attr(model_custom_update, "value") == "my-model"


def test_build_model_config_updates_custom_provider_url():
    provider_update, provider_url_update, model_choice_update, model_custom_update = (
        mod.build_model_config_updates(
            provider="custom",
            model_name="my-model",
            provider_url=" https://example.com/v1 ",
        )
    )

    assert _get_update_attr(provider_update, "value") == "custom"
    assert _get_update_attr(provider_url_update, "visible") is True
    assert _get_update_attr(provider_url_update, "value") == "https://example.com/v1"
    assert _get_update_attr(model_choice_update, "value") == "custom"
    assert _get_update_attr(model_custom_update, "visible") is True
