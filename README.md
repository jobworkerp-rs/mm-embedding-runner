# mm-embedding-runner

A multimodal embedding plugin that maps text and images into the **same vector space**. It can be used for cross-modal search and similarity comparison between text and images.

This runs as a V2 plugin (`.so` shared library) for [jobworkerp-rs](https://github.com/jobworkerp-rs/jobworkerp-rs). It is not a standalone server or CLI; load it from a jobworkerp host.

- Backend: [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) (2048 dimensions, Apache 2.0), executed through [fastembed-rs](https://crates.io/crates/fastembed)
- Inference devices: CPU / CUDA / Metal (candle backend)

## Features

| Method (`using`) | Input | Purpose |
|---|---|---|
| `embed_text` | Text | Text embeddings, with automatic chunking for long input |
| `embed_image` | Images (bytes / file paths / URLs) | Image embeddings |

Both methods share the **same model instance** internally, so their output vectors can be compared directly. Outputs are always L2-normalized (norm ~= 1).

## Protobuf Definitions

The `.proto` files under `protobuf/` are the authoritative definitions for plugin settings, method arguments, and method output.

| Purpose | Protobuf | Summary |
|---|---|---|
| Load-time settings | [`protobuf/mm_embedding_settings.proto`](protobuf/mm_embedding_settings.proto) | `MultimodalEmbeddingRunnerSettings`: model ID, device, dtype, chunking, image input limits, and related options |
| `embed_text` arguments | [`protobuf/embed_text_args.proto`](protobuf/embed_text_args.proto) | `EmbedTextArgs`: text to embed |
| `embed_image` arguments | [`protobuf/embed_image_args.proto`](protobuf/embed_image_args.proto) | `EmbedImageArgs`: image `data` / `file_path` / `url`, plus an optional `label` |
| Shared output | [`protobuf/mm_embedding_result.proto`](protobuf/mm_embedding_result.proto) | `MmEmbeddingResult`: embedding vectors, position metadata, and model information |

## Quick Start

### 1. Build

This repository uses git submodules, so fetch them after cloning.

```bash
git clone <this-repo>
cd mm-embedding-runner
git submodule update --init --recursive

cargo build --release                  # CPU build -> target/release/libmm_embedding_runner.so
cargo build --release --features cuda   # CUDA build (separate binary)
```

### 2. Register with jobworkerp

Place the generated `libmm_embedding_runner.so` in the jobworkerp host plugin directory. The host loads it with `dlopen` and exposes two methods under the plugin name `MultimodalEmbeddingRunner`.

### 3. Load-Time Settings

Settings are supplied once when the plugin is loaded. Main fields:

| Field | Default | Description |
|---|---|---|
| `model_id` | Required | HuggingFace repository ID, for example `Qwen/Qwen3-VL-Embedding-2B` |
| `device` | `CPU` | `CPU` / `CUDA` / `METAL` |
| `max_sequence_length` | `2048` | Token limit (16 to 32768) |
| `allowed_image_dirs` | `[]` | Directories allowed for `file_path` image inputs. If empty, all `file_path` inputs are rejected |
| `image_size_limit_bytes` | 10 MiB | Compressed input size limit per image |

> Model weights are not included in the distribution. They are downloaded from HuggingFace Hub on first load. Set `HF_HOME` to control the cache location; otherwise the default is `$HOME/.cache/huggingface/hub/`.

### 4. Call Examples

Input for `embed_text` (`using = "embed_text"`):

```json
{ "text": "a cat" }
```

Input for `embed_image` (`using = "embed_image"`):

```json
{
  "items": [
    { "data": "<image bytes>", "label": "photo-1" },
    { "file_path": "/srv/images/cat.png" },
    { "url": "https://example.com/cat.png" }
  ]
}
```

Shared `MmEmbeddingResult` output:

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

`begin_position` / `end_position` are character offsets into the original text for `embed_text`, and input-array indexes for `embed_image`.

### Improve Retrieval Quality (Optional)

Qwen3-VL-Embedding is instruction-aware. For retrieval queries, prepend an instruction to `text` for better quality:

```text
Instruct: Retrieve images or text relevant to the user's query.
Query: a cat
```

## Notes and Limitations

- **`file_path` input requires `allowed_image_dirs`** to prevent arbitrary local file reads. Allowed paths are checked by path component, not by raw string prefix.
- **`url` input is not hardened against SSRF.** Use it only with input from trusted callers. Only `http` and `https` URLs are accepted.
- Loading with `device = CUDA` from a CPU-only binary returns `UnsupportedDevice` instead of falling back to CPU.
- Decoded bitmap size is checked before decoding to mitigate decompression-bomb style inputs.

## Tests

```bash
cargo test                              # unit tests; no network or GPU required

# integration / expensive tests; require model download and effectively need a GPU, #[ignore]
cargo test --features cuda --test integration_embed_text -- --ignored --test-threads=1
```

Integration-test devices are selected from build features (`--features cuda` -> CUDA, etc.). Override with `MM_TEST_DEVICE=cpu|cuda|metal`.

## Distribution

- CPU and CUDA builds are distributed as separate binaries with the same `.so` name and plugin name (CPU -> `runner/`, CUDA -> `cuda_runner/`).
- TLS uses **rustls** and does not depend on OpenSSL (`libssl` / `libcrypto`). The generated `.so` can be loaded on Linux environments without depending on the host OpenSSL version.

## License

- Crate license: **MIT**
- Backend licenses: fastembed-rs (Apache-2.0), candle (Apache-2.0 / MIT), Qwen3-VL-Embedding-2B weights (Apache 2.0)

For detailed design decisions and exhaustive field descriptions, see the comments in the `.proto` files and the repository documentation.
