# mm-embedding-runner

テキストと画像を **同一のベクトル空間** へ写像するマルチモーダル embedding プラグインです。テキスト ↔ 画像のクロスモーダル検索や類似度比較に使えます。

[jobworkerp-rs](https://github.com/jobworkerp-rs/jobworkerp-rs) の V2 プラグイン（`.so` 共有ライブラリ）として動作します。単体のサーバや CLI ではなく、jobworkerp ホストに読み込ませて利用します。

- バックエンド: [fastembed-rs](https://crates.io/crates/fastembed) 経由で Qwen3-VL / Qwen3 text-only / ModernBERT ONNX を実行
- 推論デバイス: Qwen3 は CPU / CUDA / Metal（candle backend）、ModernBERT ONNX は CPU / CUDA / CoreML（`device = METAL` 指定時）

## できること

| メソッド (`using`) | 入力 | 用途 |
|---|---|---|
| `embed_text` | テキスト | テキストの embedding（長文は自動 chunking）|
| `embed_image` | 画像（バイト列 / ファイルパス / URL）| 画像の embedding |

Qwen3-VL では両メソッドが内部で**同一のモデルインスタンスを共有**するため、出力ベクトルは直接比較できます。Qwen3 text-only と ModernBERT ONNX は `embed_text` のみ対応し、`embed_image` は `InvalidArgument` になります。出力は常に L2 正規化済み（ノルム ≈ 1）です。

## Protobuf 定義

plugin の設定・各 method の引数・出力は `protobuf/` 配下の `.proto` が正です。

| 用途 | Protobuf | 概要 |
|---|---|---|
| ロード時設定 | [`protobuf/mm_embedding_settings.proto`](protobuf/mm_embedding_settings.proto) | `MultimodalEmbeddingRunnerSettings`: モデル ID、device、dtype、chunking、画像入力制限など |
| `embed_text` 引数 | [`protobuf/embed_text_args.proto`](protobuf/embed_text_args.proto) | `EmbedTextArgs`: embedding するテキスト |
| `embed_image` 引数 | [`protobuf/embed_image_args.proto`](protobuf/embed_image_args.proto) | `EmbedImageArgs`: 画像の `data` / `file_path` / `url` と任意の `label` |
| 共通出力 | [`protobuf/mm_embedding_result.proto`](protobuf/mm_embedding_result.proto) | `MmEmbeddingResult`: embedding ベクトル、位置情報、モデル情報 |

## クイックスタート

### 1. ビルド

このリポジトリは git submodule を使うため、clone 後に必ず取得してください。

```bash
git clone <this-repo>
cd mm-embedding-runner
git submodule update --init --recursive

cargo build --release                  # CPU 版 → target/release/libmm_embedding_runner.so
cargo build --release --features cuda   # CUDA 版（別バイナリ）
```

### 2. jobworkerp に登録

生成された `libmm_embedding_runner.so` を jobworkerp ホストのプラグインディレクトリに配置します。ホストが `dlopen` し、プラグイン名 `MultimodalEmbeddingRunner` として 2 つのメソッドを公開します。

### 3. ロード時の設定（Settings）

プラグインのロード時に一度だけ渡す設定です。主な項目:

| フィールド | 既定値 | 説明 |
|---|---|---|
| `model_id` | （必須）| HuggingFace リポジトリ ID。例: `Qwen/Qwen3-VL-Embedding-2B` |
| `device` | `CPU` | `CPU` / `CUDA` / `METAL` |
| `max_sequence_length` | `2048` | トークン上限（16〜32768）|
| `allowed_image_dirs` | `[]` | `file_path` 画像入力を許可するディレクトリ。**空だと `file_path` は拒否** |
| `image_size_limit_bytes` | 10 MiB | 入力画像の圧縮サイズ上限 |
| `onnx_model_file` | `onnx/model.onnx` | `ModernBertModel` の ONNX ファイル。例: `onnx/model_int8.onnx` |
| `onnx_pooling` | `ONNX_POOLING_MEAN` | ModernBERT ONNX の pooling。`ONNX_POOLING_MEAN` / `ONNX_POOLING_CLS` |

> モデル重みは配布物に含まれず、初回ロード時に HuggingFace Hub から取得されます。キャッシュ場所は環境変数 `HF_HOME` で制御できます（未設定時は `$HOME/.cache/huggingface/hub/`）。

### ModernBERT ONNX を使う

`config.json` の `architectures[0]` が `ModernBertModel` のリポジトリは、ONNX 版としてロードします。リポジトリには `config.json`、`tokenizer.json`、`special_tokens_map.json`、`tokenizer_config.json`、および `onnx_model_file` で指定した ONNX ファイルが必要です。

例:

```json
{
  "model_id": "sirasagi62/ruri-v3-310m-ONNX",
  "device": "CPU",
  "max_sequence_length": 512,
  "onnx_model_file": "onnx/model.onnx",
  "onnx_pooling": "ONNX_POOLING_MEAN"
}
```

ModernBERT ONNX の既定 pooling は mean です。CLS pooling が必要な export では `onnx_pooling` に `ONNX_POOLING_CLS` を指定してください。Ruri など prefix が推奨されるモデルでは、`検索クエリ:` や passage 用 prefix は呼び出し側で `text` に含めてください。

`model_info.dtype_used` は Qwen3 では `F32` / `F16` / `BF16`、ONNX では選択したファイル名から推定した `ONNX` / `INT8` / `F16` / `Q4` / `BNB4` などを返します。ModernBERT ONNX の `embedding_dimension` はロード時の probe 推論で得た実出力長です。

ONNX の GPU 実行は Qwen3 の candle feature とは別です。

```bash
cargo build --release --features onnx-cuda    # device = CUDA で ONNX Runtime CUDA EP
cargo build --release --features onnx-coreml  # device = METAL で ONNX Runtime CoreML EP
```

`device = METAL` は ONNX では Metal EP ではなく CoreML EP を使います。CoreML が対象 ONNX graph を受けられない場合はロード時にエラーになります。

### 4. 呼び出し例

`embed_text`（`using = "embed_text"`）の入力:

```json
{ "text": "a cat" }
```

`embed_image`（`using = "embed_image"`）の入力:

```json
{
  "items": [
    { "data": "<image bytes>", "label": "photo-1" },
    { "file_path": "/srv/images/cat.png" },
    { "url": "https://example.com/cat.png" }
  ]
}
```

両メソッド共通の出力 `MmEmbeddingResult`:

```json
{
  "embeddings": [
    { "values": [/* 2048 floats */], "begin_position": 0, "end_position": 5, "content": "a cat" }
  ],
  "model_info": {
    "model_name": "Qwen/Qwen3-VL-Embedding-2B",
    "embedding_dimension": 2048,
    "supports_vision": true
  }
}
```

`begin_position` / `end_position` は、`embed_text` では元テキストの**文字オフセット**、`embed_image` では**入力配列のインデックス**を表します。

### 検索品質を上げる（任意）

Qwen3-VL-Embedding は Instruction Aware です。検索クエリ用途では、`text` の先頭に instruction を含めると精度が上がります:

```text
Instruct: Retrieve images or text relevant to the user's query.
Query: a cat
```

## 注意・制限

- **`file_path` 入力には `allowed_image_dirs` の設定が必須**です（任意ローカルファイル読み取りを防ぐガード）。許可判定はパスのコンポーネント単位で行われます。
- **`url` 入力は SSRF 対策をしていません。** 信頼できる呼び出し元からの入力にのみ使ってください。`http` / `https` のみ受理します。
- `device = CUDA` を CPU 版バイナリでロードすると、CPU へフォールバックせず `UnsupportedDevice` エラーになります（意図した挙動）。
- 巨大画像による DoS（decompression bomb）対策として、decode 前にビットマップサイズを検査します。

## テスト

```bash
cargo test                              # unit テスト（ネットワーク・GPU 不要）

# integration / expensive テスト（モデル DL と GPU が事実上必要、#[ignore] 付き）
cargo test --features cuda --test integration_embed_text -- --ignored --test-threads=1
cargo test --test integration_modernbert_onnx -- --ignored --test-threads=1
```

integration テストの推論デバイスはビルド時の feature から自動選択されます（`--features cuda` → CUDA など）。`MM_TEST_DEVICE=cpu|cuda|metal` で上書きできます。

## 配布について

- CPU 版と CUDA 版は同じ `.so` 名・同じプラグイン名で別バイナリとして配布します（CPU 版 → `runner/`、CUDA 版 → `cuda_runner/`）。
- TLS には **rustls** を使用し、OpenSSL（`libssl` / `libcrypto`）には依存しません。生成される `.so` はホストの OpenSSL バージョンに依存せず任意の Linux 環境で `dlopen` できます。

## ライセンス

- crate ライセンス: **MIT**
- バックエンド: fastembed-rs（Apache-2.0）、candle（Apache-2.0 / MIT）、Qwen3-VL-Embedding-2B 重み（Apache 2.0）

設計判断の詳細・各設定の網羅的な説明は、`.proto` ファイルのコメントおよびリポジトリ内ドキュメントを参照してください。
