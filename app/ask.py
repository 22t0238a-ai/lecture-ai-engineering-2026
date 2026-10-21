"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保って続けて質問する

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import csv
import statistics
import sys
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
RETRYABLE_API_CODES = {500, 502, 503, 504}


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    _print_usage(response)
    return response.text


def _print_usage(response) -> None:
    input_tokens, output_tokens = _usage_counts(response)
    if input_tokens is None and output_tokens is None:
        print("[トークン] 利用量情報なし")
    else:
        print(
            f"[トークン] 入力: {input_tokens}, "
            f"出力: {output_tokens}"
        )


def _usage_counts(response) -> tuple[int | None, int | None]:
    usage = response.usage_metadata
    if usage is None:
        return None, None
    return usage.prompt_token_count, usage.candidates_token_count


def chat(initial_question: str | None = None) -> None:
    client = genai.Client(api_key=get_api_key())
    conversation = client.chats.create(model=MODEL)

    def send(question: str) -> None:
        response = conversation.send_message(question)
        print(f"[質問] {question}\n")
        _print_usage(response)
        print(response.text)

    if initial_question:
        send(initial_question)

    while True:
        try:
            question = input("質問 (終了: quit / 終了 / 空 Enter): ").strip()
        except EOFError:
            break
        if not question or question.lower() in {"quit", "exit"} or question == "終了":
            break
        send(question)


def benchmark(
    models: list[str],
    questions: list[str],
    delay_seconds: float,
    repeats: int = 1,
    continue_on_error: bool = False,
    retries: int = 0,
    retry_delay_seconds: float = 30.0,
) -> bool:
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        [
            "model",
            "question",
            "trial",
            "elapsed_seconds",
            "request_seconds",
            "retry_wait_seconds",
            "retries",
            "input_tokens",
            "output_tokens",
            "status",
        ]
    )
    client = genai.Client(api_key=get_api_key())
    total = len(models) * len(questions) * repeats
    completed = 0
    results: dict[tuple[str, str], list[dict]] = {}
    all_succeeded = True
    stop = False

    for model in models:
        for question in questions:
            key = (model, question)
            results[key] = []
            for trial in range(1, repeats + 1):
                api_error: tuple[int, str] | None = None
                retry_count = 0
                retry_wait_seconds = 0.0
                request_seconds = 0.0
                while True:
                    request_started = time.perf_counter()
                    try:
                        response = client.models.generate_content(model=model, contents=question)
                        request_seconds += time.perf_counter() - request_started
                        input_tokens, output_tokens = _usage_counts(response)
                        status = "OK_RETRIED" if retry_count else "OK"
                        api_error = None
                        break
                    except errors.APIError as e:
                        request_seconds += time.perf_counter() - request_started
                        api_error = e.code, e.message
                        if e.code in RETRYABLE_API_CODES and retry_count < retries:
                            wait_seconds = retry_delay_seconds * (2**retry_count)
                            print(
                                f"一時的なAPIエラー（{e.code}）。{wait_seconds:g}秒後に再試行します。",
                                file=sys.stderr,
                            )
                            if wait_seconds:
                                time.sleep(wait_seconds)
                            retry_wait_seconds += wait_seconds
                            retry_count += 1
                            continue
                        input_tokens, output_tokens = None, None
                        status = f"API_ERROR_{e.code}"
                        break
                elapsed = request_seconds + retry_wait_seconds

                result = {
                    "trial": trial,
                    "elapsed_seconds": elapsed,
                    "request_seconds": request_seconds,
                    "retry_wait_seconds": retry_wait_seconds,
                    "retries": retry_count,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "status": status,
                }
                results[key].append(result)
                completed += 1
                writer.writerow(
                    [
                        model,
                        question,
                        trial,
                        f"{elapsed:.3f}",
                        f"{request_seconds:.3f}",
                        f"{retry_wait_seconds:.3f}",
                        retry_count,
                        input_tokens,
                        output_tokens,
                        status,
                    ]
                )

                if api_error is not None:
                    all_succeeded = False
                    print(api_error_message(*api_error), file=sys.stderr)
                    code = api_error[0]
                    should_stop = code == 429 or 400 <= code < 500 or not continue_on_error
                    if should_stop:
                        print("APIエラーのため、残りの比較を停止しました。", file=sys.stderr)
                        stop = True
                        break

                if completed < total and delay_seconds:
                    time.sleep(delay_seconds)
            if stop:
                break
        if stop:
            break

    _print_benchmark_summary(results)
    return all_succeeded


def _print_benchmark_summary(results: dict[tuple[str, str], list[dict]]) -> None:
    print("\n[集計] 成功試行を対象に応答時間の中央値と平均トークン数を算出", file=sys.stderr)
    for (model, question), attempts in results.items():
        successful = [result for result in attempts if result["status"] == "OK"]
        successful.extend(result for result in attempts if result["status"] == "OK_RETRIED")
        failed = len(attempts) - len(successful)
        retry_total = sum(result["retries"] for result in attempts)
        elapsed = [result["elapsed_seconds"] for result in successful]
        request_seconds = [result["request_seconds"] for result in successful]
        no_retry_seconds = [
            result["request_seconds"] for result in successful if result["retries"] == 0
        ]
        input_tokens = [result["input_tokens"] for result in successful]
        output_tokens = [result["output_tokens"] for result in successful]
        input_tokens = [value for value in input_tokens if value is not None]
        output_tokens = [value for value in output_tokens if value is not None]

        median_elapsed = f"{statistics.median(elapsed):.3f}" if elapsed else "N/A"
        median_request = f"{statistics.median(request_seconds):.3f}" if request_seconds else "N/A"
        median_no_retry = f"{statistics.median(no_retry_seconds):.3f}" if no_retry_seconds else "N/A"
        mean_input = f"{statistics.mean(input_tokens):.1f}" if input_tokens else "N/A"
        mean_output = f"{statistics.mean(output_tokens):.1f}" if output_tokens else "N/A"
        total_input = sum(input_tokens) if input_tokens else 0
        total_output = sum(output_tokens) if output_tokens else 0
        print(
            f"model={model} question={question} attempts={len(attempts)} "
            f"successes={len(successful)} failures={failed} "
            f"retries={retry_total} api_calls={len(attempts) + retry_total} "
            f"median_elapsed_seconds={median_elapsed} "
            f"median_request_seconds={median_request} "
            f"median_no_retry_seconds={median_no_retry} "
            f"mean_input_tokens={mean_input} "
            f"mean_output_tokens={mean_output} "
            f"total_input_tokens={total_input} total_output_tokens={total_output}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--chat", action="store_true", help="会話履歴を保って続けて質問する")
    mode.add_argument("--benchmark", action="store_true", help="複数モデル・質問を比較する")
    parser.add_argument("--models", nargs="+", help="比較するモデル名")
    parser.add_argument("--questions", nargs="+", help="比較する質問")
    parser.add_argument("--delay", type=float, default=1.0, help="呼び出し間隔（秒、既定: 1）")
    parser.add_argument("--repeats", type=int, default=1, help="各モデル・質問の試行回数")
    parser.add_argument("--retries", type=int, default=0, help="503等の一時的な5xxへの再試行回数")
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=30.0,
        help="再試行までの初期待機秒数。再試行ごとに倍増します",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="429以外のAPIエラーを記録して次の試行へ進む",
    )
    args = parser.parse_args()

    try:
        if args.benchmark:
            if args.question or not args.models or not args.questions:
                parser.error("--benchmark には --models と --questions を指定してください")
            if args.delay < 0:
                parser.error("--delay は0以上を指定してください")
            if args.repeats < 1:
                parser.error("--repeats は1以上を指定してください")
            if args.retries < 0:
                parser.error("--retries は0以上を指定してください")
            if args.retry_delay < 0:
                parser.error("--retry-delay は0以上を指定してください")
            if not benchmark(
                args.models,
                args.questions,
                args.delay,
                args.repeats,
                args.continue_on_error,
                args.retries,
                args.retry_delay,
            ):
                raise SystemExit(1)
        elif (
            args.models
            or args.questions
            or args.repeats != 1
            or args.retries != 0
            or args.retry_delay != 30.0
            or args.continue_on_error
        ):
            parser.error("ベンチマーク用オプションは --benchmark と一緒に指定してください")
        elif args.chat:
            print(f"[model] {MODEL}")
            chat(args.question)
        else:
            print(f"[model] {MODEL}")
            question = args.question or DEFAULT_QUESTION
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
