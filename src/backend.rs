//! Lazy fastembed backend wrapper.
//!
//! The backend is constructed in `load()` (specification.md §4.4) so the
//! plugin can be discovered and have its metadata queried without touching
//! CUDA / candle. Once loaded, the same `Qwen3VLEmbedding` instance is
//! shared by both methods (`embed_text` and `embed_image`) to guarantee
//! vector-space alignment by construction (§2.2 / §4.1).

use std::sync::Arc;

use candle_core::{DType as CandleDType, Device as CandleDevice};
use fastembed::Qwen3VLEmbedding;

use crate::error::{MmEmbeddingError, Result};
use crate::settings::{DTypeKind, DeviceKind, ValidatedSettings};
use crate::tokenization::TokenizationProcessor;

pub struct MmBackend {
    model: Qwen3VLEmbedding,
    tokenizer: Arc<TokenizationProcessor>,
    settings: ValidatedSettings,
}

impl std::fmt::Debug for MmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmBackend")
            .field("model_id", &self.settings.model_id)
            .field("device", &self.settings.device)
            .field("dtype", &self.settings.dtype)
            .field("max_sequence_length", &self.settings.max_sequence_length)
            .finish_non_exhaustive()
    }
}

impl MmBackend {
    pub fn load(settings: ValidatedSettings) -> Result<Self> {
        let device = build_device(settings.device, settings.gpu_device_index)?;
        let dtype = candle_dtype(settings.dtype);

        // fastembed pulls model weights itself. We separately fetch
        // tokenizer.json for our chunking accounting so we are not coupled
        // to fastembed's private tokenizer instance.
        let tokenizer = Arc::new(TokenizationProcessor::load_from_hf(
            &settings.tokenizer_model_id,
        )?);

        let model = Qwen3VLEmbedding::from_hf(
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
        })?;

        Ok(Self {
            model,
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

    /// Batch text embedding. Takes `&[&str]` so callers can pass borrowed
    /// chunk content without cloning it into owned `String`s.
    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed_texts(texts)
            .map_err(|e| MmEmbeddingError::backend(format!("embed_texts failed: {e}")))
    }

    /// Batch image embedding from raw bytes. We never call fastembed's
    /// path-based `embed_images` because it would re-open the file after our
    /// validation step, opening a TOCTOU window (specification.md §3.5).
    pub fn embed_image_bytes(&self, images: &[&[u8]]) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed_image_bytes(images)
            .map_err(|e| MmEmbeddingError::backend(format!("embed_image_bytes failed: {e}")))
    }
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
    fn metal_without_feature_is_unsupported() {
        if cfg!(feature = "metal") {
            return;
        }
        let err = build_device(DeviceKind::Metal, 0).unwrap_err();
        assert!(err.to_string().starts_with("UnsupportedDevice:"));
    }
}
