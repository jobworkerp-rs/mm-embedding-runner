//! Lazy fastembed backend wrapper.
//!
//! The backend is constructed in `load()`, not at plugin construction, so
//! the plugin can be discovered and have its metadata queried on hosts
//! without CUDA / candle available.
//!
//! The wrapped model is one of three architectures, selected from the
//! target repo's `config.json::architectures[0]` (see [`crate::model_arch`]):
//!
//! - `Qwen3VLForConditionalGeneration` -> [`Qwen3VLEmbedding`], which serves
//!   both `embed_text` and `embed_image` from a single instance, so their
//!   vectors share one space and are directly comparable.
//! - `Qwen3ForCausalLM` -> [`Qwen3TextEmbedding`], text-only. Calls to
//!   `embed_image_bytes` on this variant return an `InvalidArgument:`
//!   error (no silent empty vector, no fallback).
//! - `ModernBertModel` -> ONNX [`TextEmbedding`], text-only with the same
//!   `embed_image_bytes` rejection as the Qwen3 text variant.

use std::sync::{Arc, Mutex};

use candle_core::{DType as CandleDType, Device as CandleDevice};
use fastembed::{
    InitOptionsUserDefined, Pooling, Qwen3TextEmbedding, Qwen3VLEmbedding, TextEmbedding,
    TokenizerFiles, UserDefinedEmbeddingModel,
};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::api::sync::ApiRepo;
use ort::execution_providers::ExecutionProviderDispatch;

use crate::error::{MmEmbeddingError, Result};
use crate::model_arch::{ArchitectureKind, detect_model_config, read_hf_file};
use crate::settings::{DTypeKind, DeviceKind, OnnxPoolingKind, ValidatedSettings};
use crate::tokenization::TokenizationProcessor;

// fastembed embedding structs are large; box both variants so the enum
// discriminant size is not driven by the bigger one.
enum InnerBackend {
    Qwen3Vl(Box<Qwen3VLEmbedding>),
    Qwen3Text(Box<Qwen3TextEmbedding>),
    ModernBertOnnx {
        model: Box<Mutex<TextEmbedding>>,
        embedding_dimension: usize,
        dtype_used: &'static str,
    },
}

pub struct MmBackend {
    inner: InnerBackend,
    tokenizer: Arc<TokenizationProcessor>,
    settings: ValidatedSettings,
}

impl std::fmt::Debug for MmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let arch = match self.inner {
            InnerBackend::Qwen3Vl(_) => "Qwen3Vl",
            InnerBackend::Qwen3Text(_) => "Qwen3Text",
            InnerBackend::ModernBertOnnx { .. } => "ModernBertOnnx",
        };
        f.debug_struct("MmBackend")
            .field("architecture", &arch)
            .field("model_id", &self.settings.model_id)
            .field("device", &self.settings.device)
            .field("dtype", &self.settings.dtype)
            .field("max_sequence_length", &self.settings.max_sequence_length)
            .finish_non_exhaustive()
    }
}

impl MmBackend {
    pub fn load(settings: ValidatedSettings) -> Result<Self> {
        reject_if_device_unsupported_by_all_backends(settings.device)?;

        let dtype = candle_dtype(settings.dtype);

        // fastembed pulls model weights itself; we fetch tokenizer.json
        // separately for our chunking accounting so we are not coupled to
        // fastembed's private tokenizer instance. Both share one HF_HOME cache
        // (see the vendored-fork note in Cargo.toml).
        let tokenizer = Arc::new(TokenizationProcessor::load_from_hf(
            &settings.tokenizer_model_id,
        )?);

        // hf-hub here shares the HF_HOME cache that fastembed's own
        // loader will hit for the same config.json, so the bytes are read
        // once on cold start and once from the cache.
        let api = ApiBuilder::from_env()
            .build()
            .map_err(|e| MmEmbeddingError::backend(format!("hf api init failed: {e}")))?;
        let repo = api.model(settings.model_id.clone());
        let model_config = detect_model_config(&repo)?;

        let inner = match model_config.architecture {
            ArchitectureKind::Qwen3Vl => {
                let device = build_device(settings.device, settings.gpu_device_index)?;
                InnerBackend::Qwen3Vl(Box::new(
                    Qwen3VLEmbedding::from_hf(
                        &settings.model_id,
                        &device,
                        dtype,
                        settings.max_sequence_length as usize,
                    )
                    .map_err(|e| {
                        MmEmbeddingError::backend(format!(
                            "Qwen3VLEmbedding::from_hf failed for {}: {e}",
                            settings.model_id
                        ))
                    })?,
                ))
            }
            ArchitectureKind::Qwen3Text => {
                let device = build_device(settings.device, settings.gpu_device_index)?;
                InnerBackend::Qwen3Text(Box::new(
                    Qwen3TextEmbedding::from_hf(
                        &settings.model_id,
                        &device,
                        dtype,
                        settings.max_sequence_length as usize,
                    )
                    .map_err(|e| {
                        MmEmbeddingError::backend(format!(
                            "Qwen3TextEmbedding::from_hf failed for {}: {e}",
                            settings.model_id
                        ))
                    })?,
                ))
            }
            ArchitectureKind::ModernBertOnnx => {
                let loaded = load_modernbert_onnx(&repo, &settings)?;
                InnerBackend::ModernBertOnnx {
                    model: Box::new(Mutex::new(loaded.model)),
                    embedding_dimension: loaded.embedding_dimension,
                    dtype_used: loaded.dtype_used,
                }
            }
        };

        Ok(Self {
            inner,
            tokenizer,
            settings,
        })
    }

    #[must_use]
    pub fn settings(&self) -> &ValidatedSettings {
        &self.settings
    }

    #[must_use]
    pub fn tokenizer(&self) -> Arc<TokenizationProcessor> {
        self.tokenizer.clone()
    }

    /// True when the loaded model can produce image embeddings comparable
    /// to its text embeddings (Qwen3-VL only). Text-only checkpoints
    /// (Qwen3-Embedding-*) return false; the runtime uses this to reject
    /// `embed_image` calls before any blocking work is scheduled.
    #[must_use]
    pub fn supports_vision(&self) -> bool {
        matches!(self.inner, InnerBackend::Qwen3Vl(_))
    }

    /// Output dimension of this model, taken from the loaded checkpoint's
    /// `Config::hidden_size` rather than hard-coded — so `ModelInfo`
    /// tracks the actual model across architectures and checkpoint sizes.
    #[must_use]
    pub fn embedding_dimension(&self) -> usize {
        match &self.inner {
            InnerBackend::Qwen3Vl(m) => m.config().hidden_size,
            InnerBackend::Qwen3Text(m) => m.config().hidden_size,
            InnerBackend::ModernBertOnnx {
                embedding_dimension,
                ..
            } => *embedding_dimension,
        }
    }

    #[must_use]
    pub fn dtype_used(&self) -> &'static str {
        match &self.inner {
            InnerBackend::Qwen3Vl(_) | InnerBackend::Qwen3Text(_) => match self.settings.dtype {
                DTypeKind::F32 => "F32",
                DTypeKind::F16 => "F16",
                DTypeKind::Bf16 => "BF16",
            },
            InnerBackend::ModernBertOnnx { dtype_used, .. } => dtype_used,
        }
    }

    /// Batch text embedding. Takes `&[&str]` so callers can pass borrowed
    /// chunk content without cloning it into owned `String`s.
    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match &self.inner {
            InnerBackend::Qwen3Vl(m) => m
                .embed_texts(texts)
                .map_err(|e| MmEmbeddingError::backend(format!("embed_texts failed: {e}"))),
            InnerBackend::Qwen3Text(m) => m
                .embed(texts)
                .map_err(|e| MmEmbeddingError::backend(format!("embed_texts failed: {e}"))),
            InnerBackend::ModernBertOnnx { model, .. } => {
                let mut guard = match model.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                guard
                    .embed(texts, None)
                    .map_err(|e| MmEmbeddingError::backend(format!("embed_texts failed: {e}")))
            }
        }
    }

    /// Batch image embedding from raw bytes. We never call fastembed's
    /// path-based `embed_images` because it would re-open the file after our
    /// validation step, opening a TOCTOU window; passing the already-read
    /// bytes closes that window.
    ///
    /// Returns `InvalidArgument:` for text-only backends, preserving the
    /// capability check at the backend boundary as well as upstream.
    pub fn embed_image_bytes(&self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        match &self.inner {
            InnerBackend::Qwen3Vl(m) => m
                .embed_image_bytes(images)
                .map_err(|e| MmEmbeddingError::backend(format!("embed_image_bytes failed: {e}"))),
            InnerBackend::Qwen3Text(_) => Err(MmEmbeddingError::invalid_argument(
                "embed_image is not supported for text-only model architectures \
                 (Qwen3ForCausalLM); use embed_text only",
            )),
            InnerBackend::ModernBertOnnx { .. } => {
                reject_text_only_image_embedding("ModernBertModel ONNX")
            }
        }
    }
}

fn reject_if_device_unsupported_by_all_backends(kind: DeviceKind) -> Result<()> {
    match kind {
        DeviceKind::Cpu => Ok(()),
        DeviceKind::Cuda => {
            if cfg!(feature = "cuda") || cfg!(feature = "onnx-cuda") {
                Ok(())
            } else {
                Err(MmEmbeddingError::unsupported_device(
                    "CUDA requested but this binary was built without --features cuda or --features onnx-cuda",
                ))
            }
        }
        DeviceKind::Metal => {
            if cfg!(feature = "metal") || cfg!(feature = "onnx-coreml") {
                Ok(())
            } else {
                Err(MmEmbeddingError::unsupported_device(
                    "Metal requested but this binary was built without --features metal or --features onnx-coreml",
                ))
            }
        }
    }
}

fn reject_text_only_image_embedding(architecture: &str) -> Result<Vec<Vec<f32>>> {
    Err(MmEmbeddingError::invalid_argument(format!(
        "embed_image is not supported for text-only model architectures ({architecture}); use embed_text only"
    )))
}

struct LoadedModernBertOnnx {
    model: TextEmbedding,
    embedding_dimension: usize,
    dtype_used: &'static str,
}

fn load_modernbert_onnx(
    repo: &ApiRepo,
    settings: &ValidatedSettings,
) -> Result<LoadedModernBertOnnx> {
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read_hf_file(repo, "tokenizer.json")?,
        config_file: read_hf_file(repo, "config.json")?,
        special_tokens_map_file: read_hf_file(repo, "special_tokens_map.json")?,
        tokenizer_config_file: read_hf_file(repo, "tokenizer_config.json")?,
    };
    let onnx_file = read_hf_file(repo, &settings.onnx_model_file)?;
    let model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files)
        .with_pooling(to_fastembed_pooling(settings.onnx_pooling));
    let options = InitOptionsUserDefined::default()
        .with_max_length(settings.max_sequence_length as usize)
        .with_execution_providers(onnx_execution_providers(settings)?);
    let mut model = TextEmbedding::try_new_from_user_defined(model, options).map_err(|e| {
        MmEmbeddingError::backend(format!(
            "TextEmbedding::try_new_from_user_defined failed for {}:{}: {e}",
            settings.model_id, settings.onnx_model_file
        ))
    })?;
    let embedding_dimension = probe_embedding_dimension(&mut model)?;
    Ok(LoadedModernBertOnnx {
        model,
        embedding_dimension,
        dtype_used: onnx_dtype_label(&settings.onnx_model_file),
    })
}

fn to_fastembed_pooling(pooling: OnnxPoolingKind) -> Pooling {
    match pooling {
        OnnxPoolingKind::Mean => Pooling::Mean,
        OnnxPoolingKind::Cls => Pooling::Cls,
    }
}

fn probe_embedding_dimension(model: &mut TextEmbedding) -> Result<usize> {
    let vectors = model
        .embed(["dimension probe"], None)
        .map_err(|e| MmEmbeddingError::backend(format!("probe ONNX embedding dimension: {e}")))?;
    vectors
        .first()
        .map(Vec::len)
        .ok_or_else(|| MmEmbeddingError::backend("probe ONNX embedding dimension returned no rows"))
}

fn onnx_dtype_label(path: &str) -> &'static str {
    let lower = path.to_ascii_lowercase();
    if lower.contains("int8") {
        "INT8"
    } else if lower.contains("fp16") || lower.contains("float16") {
        "F16"
    } else if lower.contains("bf16") {
        "BF16"
    } else if lower.contains("bnb4") {
        "BNB4"
    } else if lower.contains("q4") || lower.contains("int4") {
        "Q4"
    } else {
        "ONNX"
    }
}

fn onnx_execution_providers(
    settings: &ValidatedSettings,
) -> Result<Vec<ExecutionProviderDispatch>> {
    match settings.device {
        DeviceKind::Cpu => Ok(Vec::new()),
        DeviceKind::Cuda => build_onnx_cuda_execution_providers(settings.gpu_device_index),
        DeviceKind::Metal => build_onnx_coreml_execution_providers(),
    }
}

#[cfg(feature = "onnx-cuda")]
fn build_onnx_cuda_execution_providers(gpu_index: u32) -> Result<Vec<ExecutionProviderDispatch>> {
    Ok(vec![
        ort::ep::CUDA::default()
            .with_device_id(gpu_index as i32)
            .build()
            .error_on_failure(),
    ])
}

#[cfg(not(feature = "onnx-cuda"))]
fn build_onnx_cuda_execution_providers(_gpu_index: u32) -> Result<Vec<ExecutionProviderDispatch>> {
    Err(MmEmbeddingError::unsupported_device(
        "ONNX CUDA requested but this binary was built without --features onnx-cuda",
    ))
}

#[cfg(feature = "onnx-coreml")]
fn build_onnx_coreml_execution_providers() -> Result<Vec<ExecutionProviderDispatch>> {
    Ok(vec![ort::ep::CoreML::default().build().error_on_failure()])
}

#[cfg(not(feature = "onnx-coreml"))]
fn build_onnx_coreml_execution_providers() -> Result<Vec<ExecutionProviderDispatch>> {
    Err(MmEmbeddingError::unsupported_device(
        "ONNX CoreML requested by device=METAL but this binary was built without --features onnx-coreml",
    ))
}

fn build_device(kind: DeviceKind, gpu_index: u32) -> Result<CandleDevice> {
    match kind {
        DeviceKind::Cpu => Ok(CandleDevice::Cpu),
        DeviceKind::Cuda => {
            // Without the `cuda` feature on this crate, candle-core was built
            // without CUDA support and Device::new_cuda is a stub that returns
            // an error. Surface that as UnsupportedDevice so the caller can
            // distinguish "feature missing" from "GPU init failed".
            if !cfg!(feature = "cuda") {
                return Err(MmEmbeddingError::unsupported_device(
                    "CUDA requested but this binary was built without --features cuda",
                ));
            }
            CandleDevice::new_cuda(gpu_index as usize).map_err(|e| {
                MmEmbeddingError::backend(format!(
                    "CUDA device init failed (index={gpu_index}): {e}"
                ))
            })
        }
        DeviceKind::Metal => {
            if !cfg!(feature = "metal") {
                return Err(MmEmbeddingError::unsupported_device(
                    "Metal requested but this binary was built without --features metal",
                ));
            }
            CandleDevice::new_metal(gpu_index as usize)
                .map_err(|e| MmEmbeddingError::backend(format!("Metal device init failed: {e}")))
        }
    }
}

fn candle_dtype(kind: DTypeKind) -> CandleDType {
    match kind {
        DTypeKind::F32 => CandleDType::F32,
        DTypeKind::F16 => CandleDType::F16,
        DTypeKind::Bf16 => CandleDType::BF16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{
        DTypeKind, DecodedImageBytesLimit, DeviceKind, ResolvedChunkingConfig, ValidatedSettings,
    };

    fn cuda_request() -> ValidatedSettings {
        ValidatedSettings {
            model_id: "ignored".into(),
            device: DeviceKind::Cuda,
            gpu_device_index: 0,
            dtype: DTypeKind::F32,
            max_sequence_length: 2048,
            image_size_limit_bytes: 1024,
            max_decoded_image_bytes: DecodedImageBytesLimit::Default,
            allowed_image_dirs: vec![],
            tokenizer_model_id: "ignored".into(),
            onnx_model_file: "onnx/model.onnx".into(),
            onnx_pooling: OnnxPoolingKind::Mean,
            chunking_config: ResolvedChunkingConfig {
                max_chunk_tokens: 2048,
                min_chunk_tokens: 0,
            },
        }
    }

    #[test]
    fn cuda_without_feature_is_unsupported() {
        if cfg!(feature = "cuda") {
            // When the cuda feature is enabled this test is meaningless;
            // skip rather than asserting the opposite to keep CI on either
            // feature configuration happy.
            return;
        }
        let s = cuda_request();
        let err = build_device(s.device, s.gpu_device_index).unwrap_err();
        assert!(err.to_string().starts_with("UnsupportedDevice:"));
    }

    #[test]
    fn cuda_without_any_backend_feature_is_rejected_before_hf_access() {
        if cfg!(feature = "cuda") || cfg!(feature = "onnx-cuda") {
            return;
        }
        let err = reject_if_device_unsupported_by_all_backends(DeviceKind::Cuda).unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with("UnsupportedDevice:"), "got: {msg}");
        assert!(msg.contains("cuda"), "got: {msg}");
        assert!(msg.contains("onnx-cuda"), "got: {msg}");
    }

    #[test]
    fn metal_without_feature_is_unsupported() {
        if cfg!(feature = "metal") {
            return;
        }
        let err = build_device(DeviceKind::Metal, 0).unwrap_err();
        assert!(err.to_string().starts_with("UnsupportedDevice:"));
    }

    #[test]
    fn metal_without_any_backend_feature_is_rejected_before_hf_access() {
        if cfg!(feature = "metal") || cfg!(feature = "onnx-coreml") {
            return;
        }
        let err = reject_if_device_unsupported_by_all_backends(DeviceKind::Metal).unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with("UnsupportedDevice:"), "got: {msg}");
        assert!(msg.contains("metal"), "got: {msg}");
        assert!(msg.contains("onnx-coreml"), "got: {msg}");
    }

    #[test]
    fn cpu_is_always_allowed_before_hf_access() {
        reject_if_device_unsupported_by_all_backends(DeviceKind::Cpu).unwrap();
    }

    #[test]
    fn onnx_cuda_without_feature_is_unsupported() {
        if cfg!(feature = "onnx-cuda") {
            return;
        }
        let s = cuda_request();
        let err = onnx_execution_providers(&s).unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with("UnsupportedDevice:"), "got: {msg}");
        assert!(msg.contains("onnx-cuda"), "got: {msg}");
    }

    #[test]
    fn onnx_coreml_without_feature_is_unsupported() {
        if cfg!(feature = "onnx-coreml") {
            return;
        }
        let mut s = cuda_request();
        s.device = DeviceKind::Metal;
        let err = onnx_execution_providers(&s).unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with("UnsupportedDevice:"), "got: {msg}");
        assert!(msg.contains("onnx-coreml"), "got: {msg}");
    }

    #[test]
    fn onnx_dtype_label_is_derived_from_model_file_name() {
        assert_eq!(onnx_dtype_label("onnx/model.onnx"), "ONNX");
        assert_eq!(onnx_dtype_label("onnx/model_fp16.onnx"), "F16");
        assert_eq!(onnx_dtype_label("onnx/model_int8.onnx"), "INT8");
        assert_eq!(onnx_dtype_label("onnx/model_q4.onnx"), "Q4");
        assert_eq!(onnx_dtype_label("onnx/model_bnb4.onnx"), "BNB4");
    }

    #[test]
    fn onnx_pooling_maps_to_fastembed_pooling() {
        assert_eq!(to_fastembed_pooling(OnnxPoolingKind::Mean), Pooling::Mean);
        assert_eq!(to_fastembed_pooling(OnnxPoolingKind::Cls), Pooling::Cls);
    }

    #[test]
    fn modernbert_text_only_image_rejection_is_invalid_argument() {
        let err = reject_text_only_image_embedding("ModernBertModel ONNX").unwrap_err();
        let msg = err.to_string();
        assert!(msg.starts_with("InvalidArgument:"), "got: {msg}");
        assert!(msg.contains("ModernBertModel ONNX"), "got: {msg}");
    }
}
