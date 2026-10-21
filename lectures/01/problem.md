# 第1回 演習・宿題：開発環境を整え、LLMに質問する

## この回のゴール

- 自分の fork を Codespaces で開き、Gemini API を呼び出せる
- GitHub Copilot でコードを少し直せる
- 作業ブランチから PR を出して提出できる（以降の回もこの流れで提出する）

## 準備するもの

- GitHub の個人アカウント（会社が管理するアカウントは、組織の設定で Codespaces や Copilot が使えないことがあります）
- Google の個人アカウント（会社の Google Workspace アカウントは、管理者の設定で AI Studio が使えないことがあります）

## 演習（45分）

### 1. 講座 repo を fork する

講座 repo の右上の **Fork** から、自分のアカウントに fork を作ります。以降の作業はすべて自分の fork で行います。

### 2. Gemini API キーを発行する

[Google AI Studio](https://aistudio.google.com/apikey) で API キーを作成し、コピーします。

> APIキーは秘密情報です。コードやチャット、コミットに貼らないでください。漏れた場合は AI Studio でキーを削除し、作り直してください。

### 3. Codespaces secret に登録する

GitHub の **Settings → Codespaces → Secrets → New secret** で次のように登録します。

- Name: `GEMINI_API_KEY`
- Value: 手順2のキー
- Repository access: **自分の fork** を選ぶ

> Codespaces を起動する前に登録してください。起動後に登録した場合は、Codespaces の再起動が必要です。

### 4. Codespaces を起動する

自分の fork で **Code → Codespaces → Create codespace on main**。初回は環境の準備に数分かかります。ターミナルで次を実行し、テストが通ることを確認します。

```bash
uv run pytest -q
```

### 5. 素のLLMに質問する

```bash
uv run python -m app.ask "LoRAとは何ですか？"
uv run python -m app.ask "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
```

1つ目は一般的な用語なので、おおむね正しい答えが返ります。2つ目は講座に固有の「宿題の3レベル」について質問します。素のLLMは講座のことを知らないため、「分からない」と答えるか、一般論でもっともらしく答えます。正しい内容は `data/glossary.json` の `homework-levels` にあります。モデルの答えと比べてみましょう。

余裕があれば、ほかの講座に固有の質問も試してください。同じ質問でも実行ごとに答えが変わること、知らないことでも確信を持って答えることがあることに注目しましょう。

```bash
uv run python -m app.ask "AIE2026の講座で使う「lecNN-start タグ」とは何ですか？"
uv run python -m app.ask "AIE2026の第2回では何を学びますか？"
```

### 6. Copilot でトークン数を表示させる

`app/ask.py` の `TODO(第1回 演習)` の場所で、入力トークン数と出力トークン数を表示するよう改修します。Copilot の補完やチャットを使ってかまいません。提案されたコードは、読んで理解してから採用してください。

### 7. ブランチを作って PR を出す

```bash
git switch -c lec01
# lectures/01/answer.md を書く
git add -A
git commit -m "第1回 演習"
git push -u origin lec01
```

GitHub の自分の fork で、`lec01` → `main` の PR を作成します。**base repository が自分の fork になっていることを確認してください**（講座 repo に PR を出さないこと）。

### 8. Codespaces を停止する

左下の「Codespaces」→ **Stop Current Codespace**。起動したままだと無料枠を消費し続けます。

## 宿題（どれか1つで完了）

| レベル | 内容 |
|---|---|
| 基礎 | 同じ質問を3回ずつ、モデルを変えて（`AIE_MODEL` 環境変数）、または temperature を変えて実行し、回答の違いを answer.md にまとめる |
| 実装 | `app/ask.py` を改修し、会話履歴を保ったまま続けて質問できるようにする |
| 発展 | 応答時間とトークン数を計測し、複数のモデルや質問で比較する。無料枠のレート制限との関係も考察する |

モデルの切り替え例：

```bash
AIE_MODEL=gemini-3.5-flash uv run python -m app.ask "LoRAとは何ですか？"
```

複数のモデルと質問を比較する場合は、結果をCSVで出力できます。`--repeats` で同じ条件の試行回数を指定すると、標準エラーに成功数・失敗数、再試行回数、API呼び出し数、各種時間の中央値、トークン数の平均と合計が表示されます。CSVには総経過時間、API呼び出し時間、再試行待ち時間が別々に記録されます。
`--models` には、Google AI Studioで利用可能なモデルIDを追加できます。503など一時的な5xxエラーを再試行する場合は `--retries` を指定します。待機時間は `--retry-delay` を初期値として再試行ごとに倍増します。上限まで再試行しても失敗した場合に次の試行へ進むには `--continue-on-error` を指定します。429やその他の4xxエラーでは、利用枠や設定を保護するため常に停止します。質問は難易度を分け、同じレベルの質問に対するモデルの結果を比べます。

```bash
uv run python -m app.ask --benchmark \
	--models gemini-3.5-flash-lite gemini-3.5-flash \
	--questions \
		"基礎: LoRAとは何ですか？" \
		"中級: LoRAとフルファインチューニングでは、学習するパラメータ、必要なGPUメモリ、成果物のサイズがどう違いますか？" \
		"上級: GPUメモリが限られた環境でFAQモデルを追加学習します。LoRAとQLoRAを比較し、ランクrの選び方、メモリ・学習速度・出力品質のトレードオフを説明したうえで、適切な方法を選ぶための評価実験を提案してください。" \
	--repeats 3 --retries 1 --retry-delay 30 \
	--continue-on-error --delay 5 \
	> benchmark.csv
```

CSVには各試行のモデル、質問、総経過時間、API呼び出し時間、再試行待ち時間、トークン数、状態が記録されます。集計は標準エラーに出るため、CSVに混ざらず画面でも確認できます。再試行なしの中央値はモデル本来の応答速度に近い比較に、再試行待ち込みの中央値は実際の待ち時間の比較に使えます。質問の難易度ごとに結果を比べます。この例は再試行なしなら最大18回、1回ずつ再試行されると最大36回のAPI呼び出しになるため、利用枠を確認し、429が出たら待ってから再開してください。

temperature を変えるには、`generate_content` の `config` 引数を調べてみましょう。
