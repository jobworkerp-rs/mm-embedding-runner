#![allow(improper_ctypes_definitions)]

pub mod backend;
pub mod chunking;
pub mod error;
pub mod image_validation;
pub mod model_arch;
pub mod offset;
pub mod settings;
pub mod tokenization;

// Generated protobuf modules. The four .proto files in protobuf/ all share
// the mm_embedding package, so prost-build emits a single mm_embedding.rs.
pub mod protobuf {
    pub mod mm_embedding {
        include!(concat!(env!("OUT_DIR"), "/mm_embedding.rs"));
    }
}

use jobworkerp_client::schema_to_json_string;
use jobworkerp_plugin_abi::register_plugin_v2;
use jobworkerp_plugin_abi::v2::{CancelToken, HighLevelSink, PluginV2};
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinHandle;

use crate::backend::MmBackend;
use crate::error::{MmEmbeddingError, Result as MmResult};
use crate::settings::validate;

pub const RUNNER_NAME: &str = "MultimodalEmbeddingRunner";
pub const METHOD_EMBED_TEXT: &str = "embed_text";
pub const METHOD_EMBED_IMAGE: &str = "embed_image";

/// Owns the plugin's tokio runtime and detaches it on drop.
///
/// A multi-thread `Runtime`'s own `Drop` joins its worker threads, which is
/// blocking and panics when it runs inside an async context — and every
/// `PluginV2` method body is polled on the *host* runtime. Wrapping the
/// runtime so its `Drop` calls `shutdown_background` (which detaches without
/// joining) makes it safe to drop from any of `load`'s branches, so the
/// detach logic lives in one place instead of being repeated at each exit.
struct OwnedRuntime(Option<tokio::runtime::Runtime>);

impl OwnedRuntime {
    fn handle(&self) -> tokio::runtime::Handle {
        // `Some` for the whole lifetime; only `Drop` takes it.
        self.0.as_ref().expect("runtime present").handle().clone()
    }
}

impl Drop for OwnedRuntime {
    fn drop(&mut self) {
        if let Some(rt) = self.0.take() {
            rt.shutdown_background();
        }
    }
}

/// Per-load resources, kept together so the "runtime and backend are always
/// both present or both absent" invariant is expressed by the type rather
/// than by two parallel `Option`s.
struct LoadedState {
    rt: OwnedRuntime,
    // `Arc` so the backend can be moved into `spawn_blocking` closures
    // without releasing the plugin's reference.
    backend: Arc<MmBackend>,
}

/// Multimodal embedding plugin. Text and image inputs are embedded by a
/// single shared Qwen3-VL-Embedding instance, so their vectors live in the
/// same space and are directly comparable (e.g. cosine similarity).
///
/// V2 layout: the plugin owns its own multi-threaded tokio runtime (the
/// cdylib's `thread_local!` reactor is invisible to the host runtime, so the
/// async surface needs its own reactor). Inference and URL fetches run on
/// `spawn_blocking` so the high-level future can race them against
/// `CancelToken::cancelled()` and return `Cancelled:` promptly.
pub struct MmEmbeddingPlugin {
    loaded: Option<LoadedState>,
    token: Option<CancelToken>,
}

impl MmEmbeddingPlugin {
    pub fn new() -> Self {
        Self {
            loaded: None,
            token: None,
        }
    }
}

impl Default for MmEmbeddingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the plugin's tokio runtime. V2 manual §1: a multi-thread runtime
/// with at least one worker so `spawn_blocking` results can be polled while
/// the host awaits the returned FfiFuture.
fn build_plugin_runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
}

/// Initialise tracing on the plugin's own runtime, at most once per process.
///
/// `command_utils::tracing::tracing_init_from_env` builds an OTLP exporter
/// (tonic/hyper) that internally captures `Handle::current()`. Plugin method
/// bodies are polled on the *host* runtime, so initialising there would pin
/// the exporter to a runtime that does not outlive the call. Spawning on the
/// plugin runtime's `handle` pins it correctly.
///
/// `OnceLock::set` is atomic: the first concurrent `load()` wins and spawns;
/// the rest skip. The lock records "attempted", not "succeeded" — reaching no
/// OTLP collector is non-fatal and is intentionally not retried. Fire-and-
/// forget: `load` does not await this, so a slow OTLP handshake never delays
/// model loading. Note that because the subscriber is process-global and set
/// once, a later `load` that builds a fresh runtime does not re-init tracing;
/// the exporter keeps using the first runtime's handle (acceptable: the first
/// runtime is detached, not dropped, and span loss there is non-fatal).
fn init_tracing_on_handle(handle: &tokio::runtime::Handle) {
    static TRACING_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if TRACING_INIT.set(()).is_ok() {
        handle.spawn(async {
            command_utils::util::tracing::tracing_init_from_env()
                .await
                .ok();
        });
    }
}

/// A dispatched method with its args already decoded. Decoding happens
/// before the load-state check so a malformed payload is classified as the
/// caller's fault (`InvalidArgument:`) regardless of load state — preserving
/// the v1 public contract.
enum DecodedRequest {
    EmbedText(protobuf::mm_embedding::EmbedTextArgs),
    EmbedImage(protobuf::mm_embedding::EmbedImageArgs),
}

impl DecodedRequest {
    fn decode(using: Option<&str>, args: &[u8]) -> MmResult<Self> {
        match using {
            None => Err(MmEmbeddingError::invalid_argument("method not specified")),
            Some(METHOD_EMBED_TEXT) => Ok(Self::EmbedText(
                protobuf::mm_embedding::EmbedTextArgs::decode(args)?,
            )),
            Some(METHOD_EMBED_IMAGE) => Ok(Self::EmbedImage(
                protobuf::mm_embedding::EmbedImageArgs::decode(args)?,
            )),
            Some(other) => Err(MmEmbeddingError::invalid_argument(format!(
                "unknown method '{other}' (supported: embed_text, embed_image)"
            ))),
        }
    }

    fn embed(self, backend: &MmBackend) -> MmResult<Vec<u8>> {
        match self {
            Self::EmbedText(args) => embed_text_impl(backend, args),
            Self::EmbedImage(args) => embed_image_impl(backend, args),
        }
    }
}

/// Race a spawned blocking task against an optional cancellation token.
/// `cancel_msg` is the operation label used to build the `Cancelled:`
/// payload (`"run cancelled"` / `"load cancelled"`), so callers do not have
/// to post-process the returned error string.
async fn cancellable<T>(
    token: Option<CancelToken>,
    join: JoinHandle<MmResult<T>>,
    cancel_msg: &'static str,
) -> Result<T, String> {
    match token {
        // `biased;` + `join` first: when both arms are ready (the work
        // finished and a cancel arrived in the same poll), the completed
        // result wins instead of being silently dropped as `Cancelled:`.
        // Without `biased`, `tokio::select!` chooses a ready arm at random
        // and would occasionally discard a successful inference at the
        // timeout boundary.
        Some(t) => tokio::select! {
            biased;
            res = join => collect_join(res),
            _ = t.cancelled() => Err(MmEmbeddingError::cancelled(cancel_msg).to_string()),
        },
        None => collect_join(join.await),
    }
}

fn collect_join<T>(res: Result<MmResult<T>, tokio::task::JoinError>) -> Result<T, String> {
    match res {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(e.to_string()),
        // Join failures are infrastructure errors (panic / abort), not
        // caller-induced — surface them with the BackendError prefix so
        // they cluster with other backend faults.
        Err(e) => Err(format!("BackendError: blocking task join failed: {e}")),
    }
}

fn encode_non_streaming_method_schema(
    args_proto: String,
    result_proto: String,
    description: &str,
) -> Vec<u8> {
    jobworkerp_client::jobworkerp::data::MethodSchema {
        args_proto,
        result_proto,
        description: Some(description.to_string()),
        output_type: jobworkerp_client::jobworkerp::data::StreamingOutputType::NonStreaming as i32,
        require_client_stream: false,
        client_stream_data_proto: None,
    }
    .encode_to_vec()
}

fn cached_method_proto_map() -> &'static HashMap<String, Vec<u8>> {
    static CACHED: std::sync::OnceLock<HashMap<String, Vec<u8>>> = std::sync::OnceLock::new();
    CACHED.get_or_init(|| {
        let result_proto = include_str!("../protobuf/mm_embedding_result.proto").to_string();
        HashMap::from([
            (
                METHOD_EMBED_TEXT.to_string(),
                encode_non_streaming_method_schema(
                    include_str!("../protobuf/embed_text_args.proto").to_string(),
                    result_proto.clone(),
                    "Embed text input. Long text is chunked into sliding windows; each window becomes one embedding entry.",
                ),
            ),
            (
                METHOD_EMBED_IMAGE.to_string(),
                encode_non_streaming_method_schema(
                    include_str!("../protobuf/embed_image_args.proto").to_string(),
                    result_proto,
                    "Embed image inputs into the shared multimodal vector space. Returns one embedding per input image in submission order.",
                ),
            ),
        ])
    })
}

#[async_trait::async_trait]
impl PluginV2 for MmEmbeddingPlugin {
    fn name(&self) -> String {
        String::from(RUNNER_NAME)
    }

    fn description(&self) -> String {
        String::from(
            "Generate multimodal embeddings (text and image) with positional information in a shared vector space",
        )
    }

    fn runner_settings_proto(&self) -> String {
        include_str!("../protobuf/mm_embedding_settings.proto").to_string()
    }

    fn settings_schema(&self) -> String {
        schema_to_json_string!(
            protobuf::mm_embedding::MultimodalEmbeddingRunnerSettings,
            "settings_schema"
        )
    }

    fn method_proto_map(&self) -> HashMap<String, Vec<u8>> {
        cached_method_proto_map().clone()
    }

    fn set_cancellation_token(&mut self, token: CancelToken) {
        self.token = Some(token);
    }

    async fn load(&mut self, settings: Vec<u8>) -> Result<(), String> {
        // Detach any previous load's state before building anew. Dropping
        // `LoadedState` runs `OwnedRuntime::drop` → `shutdown_background`,
        // which is safe to call here in the host's async context (it never
        // joins worker threads); the backend's `Arc` drop is a refcount
        // decrement only.
        self.loaded = None;

        // `rt` is owned locally so any early `?`/error return drops it and
        // detaches the runtime via `OwnedRuntime::drop` — no explicit
        // teardown at each exit.
        let rt =
            OwnedRuntime(Some(build_plugin_runtime().map_err(|e| {
                format!("BackendError: failed to build plugin runtime: {e}")
            })?));
        let handle = rt.handle();

        init_tracing_on_handle(&handle);

        // hf-hub model download + GPU init are blocking; race against an
        // optional cancellation token. set_cancellation_token is per-job in
        // the manual, but the order between load and set_cancellation_token
        // is not contractually fixed, so honour the token if present.
        let join: JoinHandle<MmResult<MmBackend>> = handle.spawn_blocking(move || {
            let proto =
                protobuf::mm_embedding::MultimodalEmbeddingRunnerSettings::decode(&settings[..])?;
            MmBackend::load(validate(proto)?)
        });
        // On error, `rt` drops here and detaches; on success it moves into
        // `LoadedState`.
        let backend = cancellable(self.token.clone(), join, "load cancelled").await?;
        self.loaded = Some(LoadedState {
            rt,
            backend: Arc::new(backend),
        });
        Ok(())
    }

    async fn run(
        &mut self,
        args: Vec<u8>,
        metadata: HashMap<String, String>,
        using: Option<String>,
    ) -> (Result<Vec<u8>, String>, HashMap<String, String>) {
        // Decode (and validate `using`) before the load-state check so a
        // malformed payload is reported as InvalidArgument regardless of
        // load state — a client that probes input validity before loading
        // depends on this classification (v1 contract).
        let request = match DecodedRequest::decode(using.as_deref(), &args) {
            Ok(r) => r,
            Err(e) => return (Err(e.to_string()), metadata),
        };

        let Some(loaded) = self.loaded.as_ref() else {
            return (Err(MmEmbeddingError::NotLoaded.to_string()), metadata);
        };

        // Skip image validation / URL fetch when the backend cannot serve
        // embed_image at all.
        if matches!(request, DecodedRequest::EmbedImage(_)) && !loaded.backend.supports_vision() {
            return (
                Err(MmEmbeddingError::invalid_argument(
                    "embed_image is not supported by the loaded text-only model architecture; \
                     use embed_text only",
                )
                .to_string()),
                metadata,
            );
        }

        let handle = loaded.rt.handle();
        let backend = loaded.backend.clone();
        let token = self.token.clone();

        // Inference / URL fetch are synchronous and CPU/GPU-bound; offload
        // to the blocking pool so the high-level future can race them
        // against the cancellation token.
        let join: JoinHandle<MmResult<Vec<u8>>> =
            handle.spawn_blocking(move || request.embed(&backend));
        (cancellable(token, join, "run cancelled").await, metadata)
    }

    async fn run_stream(
        &mut self,
        _args: Vec<u8>,
        _metadata: HashMap<String, String>,
        _using: Option<String>,
        _output: HighLevelSink,
    ) -> Result<HashMap<String, String>, String> {
        // Both methods declare NonStreaming in `method_proto_map`, so the
        // host should never dispatch here. A misuse returns InvalidArgument
        // with the public prefix so callers can detect it the same way they
        // detect other misuse.
        Err(MmEmbeddingError::invalid_argument(
            "streaming not supported for MultimodalEmbeddingRunner methods",
        )
        .to_string())
    }
}

/// A produced embedding plus the position/content metadata that differs
/// between the two methods (char offsets for text, array index for image).
struct EmbeddingEntry {
    begin_position: u32,
    end_position: u32,
    content: String,
}

/// Shared tail of both method handlers: validate the backend returned one
/// vector per input, then assemble the result protobuf. `op` names the
/// backend call for the mismatch error message.
fn finalize_embeddings(
    backend: &MmBackend,
    entries: Vec<EmbeddingEntry>,
    vectors: Vec<Vec<f32>>,
    op: &str,
) -> MmResult<Vec<u8>> {
    if vectors.len() != entries.len() {
        return Err(MmEmbeddingError::backend(format!(
            "{op} returned {} vectors for {} inputs",
            vectors.len(),
            entries.len()
        )));
    }
    let embeddings = entries
        .into_iter()
        .zip(vectors)
        .map(
            |(e, v)| protobuf::mm_embedding::mm_embedding_result::Embedding {
                values: v,
                begin_position: e.begin_position,
                end_position: e.end_position,
                content: e.content,
            },
        )
        .collect();
    let result = protobuf::mm_embedding::MmEmbeddingResult {
        embeddings,
        model_info: Some(build_model_info(backend)),
    };
    Ok(result.encode_to_vec())
}

fn embed_text_impl(
    backend: &MmBackend,
    args: protobuf::mm_embedding::EmbedTextArgs,
) -> MmResult<Vec<u8>> {
    if args.text.trim().is_empty() {
        return Err(MmEmbeddingError::invalid_argument("text must not be empty"));
    }

    let chunks = chunking::chunk_text(
        &args.text,
        backend.tokenizer(),
        backend.settings().chunking_config,
    )?;

    let chunk_texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
    let vectors = backend.embed_texts(&chunk_texts)?;

    let entries = chunks
        .into_iter()
        .map(|c| EmbeddingEntry {
            begin_position: c.char_start as u32,
            end_position: c.char_end as u32,
            content: c.content,
        })
        .collect();
    finalize_embeddings(backend, entries, vectors, "embed_texts")
}

fn embed_image_impl(
    backend: &MmBackend,
    args: protobuf::mm_embedding::EmbedImageArgs,
) -> MmResult<Vec<u8>> {
    let prepared = image_validation::prepare_images(&args.items, backend.settings())?;

    let byte_refs: Vec<&[u8]> = prepared.iter().map(|p| p.bytes.as_slice()).collect();
    let vectors = backend.embed_image_bytes(&byte_refs)?;

    let entries = prepared
        .into_iter()
        .enumerate()
        .map(|(i, p)| EmbeddingEntry {
            // For images, position fields are the input array index (not
            // char offsets): begin = i, end = i + 1.
            begin_position: i as u32,
            end_position: (i as u32) + 1,
            content: p.source_hint,
        })
        .collect();
    finalize_embeddings(backend, entries, vectors, "embed_image_bytes")
}

fn build_model_info(backend: &MmBackend) -> protobuf::mm_embedding::ModelInfo {
    let dtype_used = match backend.settings().dtype {
        settings::DTypeKind::F32 => "F32",
        settings::DTypeKind::F16 => "F16",
        settings::DTypeKind::Bf16 => "BF16",
    };
    protobuf::mm_embedding::ModelInfo {
        model_name: backend.settings().model_id.clone(),
        embedding_dimension: backend.embedding_dimension() as u32,
        dtype_used: dtype_used.to_string(),
        supports_vision: backend.supports_vision(),
        supports_audio: false,
        audio_sample_rate: None,
    }
}

// `register_plugin_v2!` generates the `extern "C"` `load_multi_method_plugin_v2`
// entry point and the `#[repr(C)]` vtable thunks. The init expression runs
// once per plugin instance.
//
// Why no tracing init here: `command_utils::tracing::tracing_init_from_env`
// builds an OTLP exporter (`SpanExporter::builder().with_tonic().build()`),
// which constructs a hyper/tonic client that captures `Handle::current()`
// internally. The init_expr runs on the host thread with no plugin-side
// reactor; the captured handle would be the host runtime's, but the spawned
// background tasks then run on a transient `new_current_thread` runtime
// built only for this init call and dropped immediately, leaving the
// exporter pointing at a dead runtime. The same trap applies inside the
// `load()` body if it is awaited directly on the host runtime — instead,
// tracing init is funneled through the plugin's own runtime via
// `handle.spawn(...)` (see `load`).
register_plugin_v2!(MmEmbeddingPlugin, {
    dotenvy::dotenv().ok();
    MmEmbeddingPlugin::new()
});

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::task::JoinHandle;

    fn build_rt() -> tokio::runtime::Runtime {
        build_plugin_runtime().expect("test runtime")
    }

    #[test]
    fn cancellable_without_token_returns_ok() {
        let rt = build_rt();
        let join: JoinHandle<MmResult<u32>> = rt.spawn_blocking(|| Ok(42));
        let out = rt
            .block_on(cancellable(None, join, "run cancelled"))
            .expect("ok");
        assert_eq!(out, 42);
    }

    #[test]
    fn cancellable_without_token_propagates_err() {
        let rt = build_rt();
        let join: JoinHandle<MmResult<u32>> =
            rt.spawn_blocking(|| Err(MmEmbeddingError::invalid_argument("nope")));
        let err = rt
            .block_on(cancellable(None, join, "run cancelled"))
            .expect_err("err");
        assert!(err.starts_with("InvalidArgument:"), "got: {err}");
    }

    #[test]
    fn cancellable_with_pre_cancelled_token_returns_cancelled() {
        use jobworkerp_plugin_abi::cancel::FfiCancellationToken;
        let rt = build_rt();
        // A task that never completes; since the token is pre-cancelled the
        // cancelled() branch must win. `pending()` (not a blocking sleep) so
        // dropping the JoinHandle on cancellation detaches a cooperatively
        // abortable task — the runtime drop at test end does not wait for it.
        let join: JoinHandle<MmResult<u32>> =
            rt.spawn(async { std::future::pending::<MmResult<u32>>().await });
        let (ffi, handle) = FfiCancellationToken::new_owned();
        handle.cancel();
        let token = CancelToken::from_ffi(ffi);
        let err = rt
            .block_on(cancellable(Some(token), join, "run cancelled"))
            .expect_err("cancelled");
        assert!(err.starts_with("Cancelled:"), "got: {err}");
    }

    #[test]
    fn cancellable_with_uncancelled_token_returns_ok() {
        use jobworkerp_plugin_abi::cancel::FfiCancellationToken;
        let rt = build_rt();
        let join: JoinHandle<MmResult<u32>> = rt.spawn_blocking(|| Ok(7));
        let (ffi, _handle) = FfiCancellationToken::new_owned();
        let token = CancelToken::from_ffi(ffi);
        let out = rt
            .block_on(cancellable(Some(token), join, "run cancelled"))
            .expect("ok");
        assert_eq!(out, 7);
    }

    #[test]
    fn owned_runtime_drops_safely_in_async_context() {
        // Regression: a plain multi-thread `Runtime` panics when dropped
        // inside an async context (its Drop joins worker threads). `load`
        // drops the previous `LoadedState` — and thus its `OwnedRuntime` —
        // while running on the host runtime, so `OwnedRuntime::drop` must
        // detach via `shutdown_background` instead of joining. Drop one
        // here from within `block_on` to prove it does not panic.
        let outer = build_rt();
        outer.block_on(async {
            let owned = OwnedRuntime(Some(build_plugin_runtime().unwrap()));
            // Keep the runtime alive enough to hold a worker thread, then
            // drop in-context.
            let _ = owned.handle();
            drop(owned);
        });
    }

    #[test]
    fn cancellable_uses_provided_cancel_msg() {
        use jobworkerp_plugin_abi::cancel::FfiCancellationToken;
        let rt = build_rt();
        let join: JoinHandle<MmResult<u32>> =
            rt.spawn(async { std::future::pending::<MmResult<u32>>().await });
        let (ffi, handle) = FfiCancellationToken::new_owned();
        handle.cancel();
        let token = CancelToken::from_ffi(ffi);
        let err = rt
            .block_on(cancellable(Some(token), join, "load cancelled"))
            .expect_err("cancelled");
        assert!(err.contains("load cancelled"), "got: {err}");
    }
}
