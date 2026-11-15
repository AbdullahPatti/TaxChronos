import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).resolve().parent.parent))

from google.genai import errors

from pipeline import llm


def error(kind, code):
    return kind(code, {"error": {"code": code, "message": "x", "status": "x"}}, None)


def run_with(outcomes, model="primary"):
    calls = []

    def generate_content(model, contents, config):
        calls.append(model)
        outcome = outcomes[model]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    real_client, real_fallbacks = llm.client, llm.FALLBACKS
    llm.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    llm.FALLBACKS = ["second", "third"]
    try:
        return llm.generate(model, "prompt", None), calls
    except errors.APIError as raised:
        return raised, calls
    finally:
        llm.client, llm.FALLBACKS = real_client, real_fallbacks


def test_success_makes_one_call():
    assert run_with({"primary": "ok"}) == ("ok", ["primary"])


def test_quota_overload_and_retired_models_fall_through():
    outcomes = {"primary": error(errors.ClientError, 429), "second": error(errors.ServerError, 503), "third": "ok"}
    assert run_with(outcomes) == ("ok", ["primary", "second", "third"])
    assert run_with({"primary": error(errors.ClientError, 404), "second": "ok"}) == ("ok", ["primary", "second"])


def test_bad_request_does_not_fall_through():
    result, calls = run_with({"primary": error(errors.ClientError, 400)})
    assert result.code == 400 and calls == ["primary"]


def test_last_error_is_raised_when_all_fail():
    busy = error(errors.ServerError, 503)
    result, calls = run_with({"primary": busy, "second": busy, "third": busy})
    assert result is busy and calls == ["primary", "second", "third"]


def test_fallback_model_is_not_called_twice():
    result, calls = run_with({"second": error(errors.ClientError, 429), "third": "ok"}, model="second")
    assert result == "ok" and calls == ["second", "third"]


if __name__ == "__main__":
    for name, check in list(globals().items()):
        if name.startswith("test_"):
            check()
            print("ok", name)
