import pytest

from src.factory import create_chat_model


@pytest.fixture(autouse=True)
def deepseek_api_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")


def test_create_deepseek_chat_model_disables_thinking_mode_by_default():
    model = create_chat_model(
        provider="deepseek",
        model_name="deepseek-v4-flash",
    )

    assert model.extra_body == {
        "thinking": {
            "type": "disabled",
        }
    }


def test_create_deepseek_chat_model_preserves_explicit_extra_body():
    model = create_chat_model(
        provider="deepseek",
        model_name="deepseek-v4-flash",
        extra_body={
            "thinking": {
                "type": "enabled",
            },
            "custom": "value",
        },
    )

    assert model.extra_body == {
        "thinking": {
            "type": "enabled",
        },
        "custom": "value",
    }
