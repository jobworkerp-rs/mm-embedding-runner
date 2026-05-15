# mm-embedding-runner

画像とテキストを **同一の検索可能ベクトル空間** へ写像できるマルチモーダル embedding を、jobworkerp の `MultiMethodPluginRunner` プラグインとして提供する独立 crate です。

- 仕様: [`../ai-docs/mm-embedding/specification.md`](../ai-docs/mm-embedding/specification.md)（WHAT）
- 実装計画: [`../ai-docs/mm-embedding/implementation-plan.md`](../ai-docs/mm-embedding/implementation-plan.md)（HOW）

---

## 1. モデル選定理由

**主モデル: `Qwen/Qwen3-VL-Embedding-2B`**

- Qwen3-VL ベースの multimodal embedding 専用 fine-tune。ベクトル次元 2048、モデル上限 32K トークン、Instruction Aware、Apache 2.0
- バックエンドに [`fastembed-rs`](https://crates.io/crates/fastembed)（`qwen3` feature、`=5.13.4` で pin）を採用。Rust ネイティブで candle backend により CUDA/Metal が利用可能

### ベクトル空間整合性の根拠

`fastembed::Qwen3VLEmbedding` は単一構造体で `embed_texts` / `embed_image_bytes` を同時に公開しています。本ランナーは `load()` で **1 つの `Qwen3VLEmbedding` インスタンスを構築し、`embed_text` / `embed_image` 両 method がその単一インスタンスを共有** します。これにより、テキスト経路と画像経路が確実に同じ重み・同じ device・同じ dtype を使うことを構造上保証しています。

---

## 2. method 一覧

| `using` 値 | 用途 | Args | Result |
|---|---|---|---|
| `"embed_text"` | テキスト embedding（長文は chunking） | `EmbedTextArgs` | `MmEmbeddingResult` |
| `"embed_image"` | 画像 embedding | `EmbedImageArgs` | `MmEmbeddingResult` |

両 method とも非ストリーミング（`NON_STREAMING`）、client streaming 不要。`using = None` や未知の値は `InvalidArgument:` を返し、`embed_text` へのフォールバックはしません。

### JSON / proto 例

`embed_text` の入力:

```json
{ "text": "a cat" }
```

`embed_image` の入力（バイナリ／ファイルパス／ラベル）:

```json
{
  "items": [
    { "data": "<image bytes>", "label": "photo-1" },
    { "file_path": "/srv/images/cat.png" }
  ]
}
```

出力（両 method 共通の `MmEmbeddingResult`）:

```json
{
  "embeddings": [
    { "values": [/* 2048 floats */], "begin_position": 0, "end_position": 5, "content": "a cat" }
  ],
  "model_info": {
    "model_name": "Qwen/Qwen3-VL-Embedding-2B",
    "embedding_dimension": 2048,
    "dtype_used": "F32",
    "supports_vision": true,
    "supports_audio": false
  }
}
```

`MmEmbeddingResult` は既存 `embedding-llm` の `EmbeddingLlmResult` と **JSON shape 互換・protobuf wire 互換** です（型名は区別されます）。

### 画像 method の position 意味論

`embed_image` では position フィールドを「入力配列内のインデックス」に転用します（文字オフセットではありません）:

| フィールド | 意味 |
|---|---|
| `begin_position` | 入力 `items` 配列の 0-based インデックス |
| `end_position` | `begin_position + 1`（常に 1 件分） |
| `content` | `label` があればそれ、なければ `file_path`、なければ `"image[#i]"` |

`embed_text` では `begin_position` / `end_position` は **元テキストの文字数オフセット**（char index 単位）であり、`content == text.chars().skip(begin).take(end-begin).collect()` が成立します。

---

## 3. 検索クエリ向け instruction prefix の付け方

Qwen3-VL-Embedding は **Instruction Aware** です。本ランナーは `instruction` 専用フィールドを持たないため、検索品質を高めたい場合は呼び出し側で `text` に instruction prefix を直接含めてください:

```text
Instruct: Retrieve images or text relevant to the user's query.
Query: a cat
```

将来 fastembed-rs が instruction 専用 API を提供した時点で専用フィールドを追加します。

---

## 4. 出力は常に L2 正規化済み

fastembed 5.13.4 の `Qwen3VLEmbedding` はソースレベルで `l2_normalize` を強制しています。本ランナーは逆正規化手段を提供しないため、**`values` は常に L2 正規化済み（ノルム ≈ 1）** です。Settings / Args に `normalize_embeddings` フィールドはありません。

---

## 5. セキュリティと制限

### `file_path` ソースには `allowed_image_dirs` が必須

`EmbedImageArgs.items[].file_path` を使うには Settings の `allowed_image_dirs` に許可ディレクトリを登録する必要があります。**空配列（既定値）の場合 `file_path` ソースは一切受理しません。** これは任意ローカルファイル読み取りを防ぐためのガードです。

許可判定は Settings load 時に canonical 化した `PathBuf` 配列に対する `Path::starts_with`（component-aware）で行います。文字列 prefix 一致ではないため、許可 `/tmp/allow` で `/tmp/allow_evil/x.png` を許可することはありません。`["/"]` を指定すると任意パスを許可しますが本番非推奨です。

### 画像サイズ制限の 2 段構成

| 設定 | 意味 |
|---|---|
| `image_size_limit_bytes` | 圧縮された入力バイト上限（既定 10 MiB）。入力受信時の早期 DoS ガード |
| `max_decoded_image_bytes` | decode 後 bitmap サイズ上限。`image` crate の `into_dimensions()` で安価に preflight。未指定なら 256 MiB、`0` で無制限 |

**`image_size_limit_bytes` だけでは「小さい圧縮画像が巨大 bitmap に展開される」zip bomb 類似攻撃を防げません。** decode 後サイズは `width * height * 4` を checked arithmetic で計算し、overflow 時は `InvalidArgument: decoded size overflow` を返します。

### TOCTOU の保証範囲

`file_path` は検証直後に `read_to_end` でメモリに読み込み、そのバイト列を embedding に使います。よって **embed 自体への TOCTOU は回避** されます。一方、`canonicalize` の時点とファイル open の時点の間で symlink を差し替えるレースは OS レベルの問題であり本ランナー単体では完全には防げません。許可ディレクトリは信頼可能なもののみに限定するのが利用側の責務です。

### `url` ソースは v0.1 未対応

`EmbedImageArgs.items[].url` は proto 上に予約されていますが、v0.1 では `InvalidArgument: url source is not supported in v0.1` を返します。

---

## 6. モデルロード

`load()` は `Qwen3VLEmbedding::from_hf` で **HuggingFace Hub 経由のみ** モデルを取得します（v0.1 ではローカルパス直接ロード未対応）。重みをローカル保持したい場合は環境変数 `HF_HOME` / `HF_HUB_CACHE` で hf-hub のキャッシュ場所を制御してください。

char offset 計算用の tokenizer.json は fastembed とは別に hf-hub 経由で取得します（`tokenizer_model_id` 未指定時は `model_id` と同じものを使用）。

---

## 7. ビルドと配布

```bash
# CPU 版
cargo build --release

# CUDA 版（別バイナリとして配布）
cargo build --release --features cuda
```

CUDA 版は CPU 版と別の `.so` として配布し、`lib<name>_cuda.so` にリネームして `cuda_runner/` ディレクトリに配置します（既存 CLAUDE.md「CUDA Shared Library Dependency - Final Solution」を踏襲）。プラグイン名は CPU/CUDA で同一（`MultimodalEmbeddingRunner`）です。

`device = CUDA` を CUDA feature 無効のバイナリでロードすると `UnsupportedDevice:` を返します。自動 CPU フォールバックは行いません。

---

## 8. テスト

| 階層 | 実行コマンド | モデル DL |
|---|---|---|
| unit | `cargo test` | 不要（`tests/fixtures/tokenizer.json` 同梱でネットワーク非依存）|
| integration | `cargo test --test integration_tokenizer_hf --test integration_backend_load --test integration_embed_text --test integration_embed_image -- --ignored` | 要 |
| expensive | `cargo test --test expensive_cross_modal -- --ignored` | 要 |

`#[ignore]` 付きテストはネットワーク・GPU を必要とするため PR ごと CI では実行されません。

integration / expensive テストは既定で **CUDA** を使います（CPU 推論は実用にならないほど遅いため）。device は環境変数 `MM_TEST_DEVICE` で切り替えられます:

```bash
# CUDA（既定。--features cuda でビルドが必要）
cargo test --features cuda --test integration_embed_text -- --ignored --test-threads=1

# CPU（非常に遅い、動作確認用）
MM_TEST_DEVICE=cpu cargo test --test integration_embed_text -- --ignored --test-threads=1
```

---

## 9. 依存運用

`jobworkerp-client` / `command-utils` は親リポジトリの submodule を path 参照しています:

```bash
git submodule update --init --recursive
```

submodule の revision を更新したときは本 crate の `Cargo.lock` も同時更新してください。本 crate は親ワークスペース外の独立ワークスペース（`Cargo.toml` 先頭の空 `[workspace]` テーブル）であり、独自の `mm-embedding-runner/Cargo.lock` を持ちます（再現性のため版管理対象）。

---

## 10. ライセンス

- crate ライセンス: MIT
- 同梱 `tests/fixtures/tokenizer.json`: Qwen3-VL-Embedding-2B モデルカード（Apache 2.0）に従い同梱
- 依存ライセンス: fastembed-rs（Apache-2.0）、candle-core（Apache-2.0 / MIT）、Qwen3-VL-Embedding-2B 重み（Apache 2.0）

モデル重みは配布物に含めず、初回実行時に HuggingFace Hub から取得します。
