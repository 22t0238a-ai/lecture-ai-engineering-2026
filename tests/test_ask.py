from types import SimpleNamespace
import csv
from io import StringIO

import pytest

from app import ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="回答",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
        ),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("質問") == "回答"
    assert capsys.readouterr().out == "[トークン] 入力: 12, 出力: 34\n"


def test_chat_reuses_conversation_for_follow_up_questions(monkeypatch, capsys):
    questions = []

    def send_message(question):
        questions.append(question)
        return SimpleNamespace(
            text=f"{question}への回答",
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=34,
            ),
        )

    conversation = SimpleNamespace(send_message=send_message)
    client = SimpleNamespace(
        chats=SimpleNamespace(create=lambda **kwargs: conversation)
    )
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    prompts = iter(["続きの質問", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))

    ask_module.chat("最初の質問")

    assert questions == ["最初の質問", "続きの質問"]
    output = capsys.readouterr().out
    assert "最初の質問への回答" in output
    assert "続きの質問への回答" in output


def test_benchmark_compares_models_and_questions(monkeypatch, capsys):
    calls = []

    def generate_content(*, model, contents):
        calls.append((model, contents))
        return SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=34,
            ),
        )

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask_module.time,
        "perf_counter",
        iter([1.0, 1.25, 2.0, 2.5, 3.0, 3.75, 4.0, 4.5]).__next__,
    )
    delays = []
    monkeypatch.setattr(ask_module.time, "sleep", delays.append)

    ask_module.benchmark(["model-a", "model-b"], ["質問1", "質問2"], 0.5)

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert calls == [
        ("model-a", "質問1"),
        ("model-a", "質問2"),
        ("model-b", "質問1"),
        ("model-b", "質問2"),
    ]
    assert len(rows) == 4
    assert rows[0]["elapsed_seconds"] == "0.250"
    assert rows[0]["input_tokens"] == "12"
    assert rows[0]["output_tokens"] == "34"
    assert all(row["status"] == "OK" for row in rows)
    assert delays == [0.5, 0.5, 0.5]


def test_benchmark_stops_and_reports_failure_on_api_error(monkeypatch, capsys):
    calls = []

    class FakeAPIError(Exception):
        code = 503
        message = "service unavailable"

    def generate_content(*, model, contents):
        calls.append((model, contents))
        raise FakeAPIError()

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.errors, "APIError", FakeAPIError)

    succeeded = ask_module.benchmark(["model-a", "model-b"], ["質問1", "質問2"], 0)

    output = capsys.readouterr()
    rows = list(csv.DictReader(StringIO(output.out)))
    assert not succeeded
    assert calls == [("model-a", "質問1")]
    assert len(rows) == 1
    assert rows[0]["status"] == "API_ERROR_503"
    assert "残りの比較を停止しました" in output.err


def test_benchmark_repeats_and_summarizes_results(monkeypatch, capsys):
    calls = []

    def generate_content(*, model, contents):
        calls.append((model, contents))
        index = len(calls)
        return SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=index * 10,
                candidates_token_count=index * 20,
            ),
        )

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask_module.time,
        "perf_counter",
        iter([1.0, 2.0, 3.0, 5.0]).__next__,
    )

    succeeded = ask_module.benchmark(["model-a"], ["質問"], 0, repeats=2)

    output = capsys.readouterr()
    rows = list(csv.DictReader(StringIO(output.out)))
    assert succeeded
    assert len(rows) == 2
    assert [row["trial"] for row in rows] == ["1", "2"]
    assert rows[0]["elapsed_seconds"] == "1.000"
    assert (
        "successes=2 failures=0 retries=0 api_calls=2 "
        "median_elapsed_seconds=1.500 median_request_seconds=1.500 "
        "median_no_retry_seconds=1.500"
    ) in output.err
    assert "mean_input_tokens=15.0 mean_output_tokens=30.0" in output.err
    assert "total_input_tokens=30 total_output_tokens=60" in output.err


def test_benchmark_continues_after_503_only_when_requested(monkeypatch, capsys):
    calls = []

    class FakeAPIError(Exception):
        code = 503
        message = "service unavailable"

    def generate_content(*, model, contents):
        calls.append((model, contents))
        if len(calls) == 1:
            raise FakeAPIError()
        return SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=34,
            ),
        )

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.errors, "APIError", FakeAPIError)

    succeeded = ask_module.benchmark(
        ["model-a", "model-b"], ["質問"], 0, continue_on_error=True
    )

    output = capsys.readouterr()
    rows = list(csv.DictReader(StringIO(output.out)))
    assert not succeeded
    assert len(calls) == 2
    assert [row["status"] for row in rows] == ["API_ERROR_503", "OK"]
    assert "successes=0 failures=1" in output.err
    assert "successes=1 failures=0" in output.err


def test_benchmark_retries_503_and_records_retry_count(monkeypatch, capsys):
    calls = []

    class FakeAPIError(Exception):
        code = 503
        message = "service unavailable"

    def generate_content(*, model, contents):
        calls.append((model, contents))
        if len(calls) <= 2:
            raise FakeAPIError()
        return SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=34,
            ),
        )

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.errors, "APIError", FakeAPIError)
    monkeypatch.setattr(
        ask_module.time,
        "perf_counter",
        iter([1.0, 1.2, 2.0, 2.5, 3.0, 4.0]).__next__,
    )
    delays = []
    monkeypatch.setattr(ask_module.time, "sleep", delays.append)

    succeeded = ask_module.benchmark(
        ["model-a"], ["質問"], 0, retries=2, retry_delay_seconds=3
    )

    output = capsys.readouterr()
    rows = list(csv.DictReader(StringIO(output.out)))
    assert succeeded
    assert len(calls) == 3
    assert rows[0]["retries"] == "2"
    assert rows[0]["status"] == "OK_RETRIED"
    assert rows[0]["request_seconds"] == "1.700"
    assert rows[0]["retry_wait_seconds"] == "9.000"
    assert rows[0]["elapsed_seconds"] == "10.700"
    assert delays == [3, 6]
    assert (
        "successes=1 failures=0 retries=2 api_calls=3 "
        "median_elapsed_seconds=10.700 median_request_seconds=1.700 "
        "median_no_retry_seconds=N/A"
    ) in output.err


def test_benchmark_always_stops_on_429(monkeypatch, capsys):
    calls = []

    class FakeAPIError(Exception):
        code = 429
        message = "rate limited"

    def generate_content(*, model, contents):
        calls.append((model, contents))
        raise FakeAPIError()

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(ask_module.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.errors, "APIError", FakeAPIError)

    succeeded = ask_module.benchmark(
        ["model-a", "model-b"], ["質問"], 0, continue_on_error=True
    )

    output = capsys.readouterr()
    rows = list(csv.DictReader(StringIO(output.out)))
    assert not succeeded
    assert calls == [("model-a", "質問")]
    assert len(rows) == 1
    assert rows[0]["status"] == "API_ERROR_429"
    assert "レート制限に達しました" in output.err


def test_main_exits_nonzero_when_benchmark_fails(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["app.ask", "--benchmark", "--models", "model-a", "--questions", "質問"],
    )
    monkeypatch.setattr(ask_module, "benchmark", lambda *args: False)

    with pytest.raises(SystemExit) as exc_info:
        ask_module.main()

    assert exc_info.value.code == 1