//! Settings validation for the multimodal embedding runner.
//!
//! The wire-format `MultimodalEmbeddingRunnerSettings` is decoded by the
//! plugin loader; this module turns it into a `ValidatedSettings` struct
//! whose fields are guaranteed valid (model_id non-empty,
//! max_sequence_length in range, chunking bounds consistent, allowed image
//! dirs canonicalized).

use std::path::PathBuf;

use crate::error::{MmEmbeddingError, Result};
use crate::protobuf::mm_embedding::{
    ChunkingConfig as ProtoChunkingConfig, DType as ProtoDType, Device as ProtoDevice,
    MultimodalEmbeddingRunnerSettings,
};

pub const MAX_SEQ_LEN_MIN: u32 = 16;
pub const MAX_SEQ_LEN_MAX: u32 = 32_768;
pub const DEFAULT_IMAGE_SIZE_LIMIT_BYTES: u64 = 10 * 1024 * 1024;
pub const DEFAULT_MAX_DECODED_IMAGE_BYTES: u64 = 256 * 1024 * 1024;

/// Tri-state representation of the `max_decoded_image_bytes` setting:
/// - `Default` = field unset on the wire => apply 256 MiB cap
/// - `Unlimited` = field present with value 0 => skip preflight comparison
/// - `Limited(n)` = field present with non-zero value => cap at n
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodedImageBytesLimit {
    Default,
    Unlimited,
    Limited(u64),
}

impl DecodedImageBytesLimit {
    /// Effective byte cap, or `None` if unlimited (caller skips comparison).
    #[must_use]
    pub fn effective_bytes(&self) -> Option<u64> {
        match self {
            Self::Default => Some(DEFAULT_MAX_DECODED_IMAGE_BYTES),
            Self::Unlimited => None,
            Self::Limited(n) => Some(*n),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedChunkingConfig {
    pub max_chunk_tokens: u32,
    pub min_chunk_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeKind {
    F32,
    F16,
    Bf16,
}

#[derive(Debug, Clone)]
pub struct ValidatedSettings {
    pub model_id: String,
    pub device: DeviceKind,
    pub gpu_device_index: u32,
    pub dtype: DTypeKind,
    pub max_sequence_length: u32,
    pub image_size_limit_bytes: u64,
    pub max_decoded_image_bytes: DecodedImageBytesLimit,
    /// Canonicalized whitelist of directories for the file_path image source.
    pub allowed_image_dirs: Vec<PathBuf>,
    pub tokenizer_model_id: String,
    pub chunking_config: ResolvedChunkingConfig,
}

pub fn validate(settings: MultimodalEmbeddingRunnerSettings) -> Result<ValidatedSettings> {
    // proto3 cannot distinguish unset string from empty string on the wire,
    // so a trim-empty model_id must be rejected explicitly.
    let model_id = settings.model_id.trim().to_string();
    if model_id.is_empty() {
        return Err(MmEmbeddingError::invalid_argument("model_id is required"));
    }

    // max_sequence_length range check. Treat 0 (proto3 default) as "not set"
    // and apply the runner default of 2048 to mirror the documented behavior
    // even when callers omit the field.
    let max_sequence_length = if settings.max_sequence_length == 0 {
        2048
    } else {
        settings.max_sequence_length
    };
    if !(MAX_SEQ_LEN_MIN..=MAX_SEQ_LEN_MAX).contains(&max_sequence_length) {
        return Err(MmEmbeddingError::invalid_argument(format!(
            "max_sequence_length must be in [{MAX_SEQ_LEN_MIN}, {MAX_SEQ_LEN_MAX}], got {max_sequence_length}"
        )));
    }

    let chunking_config = resolve_chunking_config(settings.chunking_config, max_sequence_length)?;

    let device = match ProtoDevice::try_from(settings.device).unwrap_or(ProtoDevice::Unspecified) {
        ProtoDevice::Cuda => DeviceKind::Cuda,
        ProtoDevice::Metal => DeviceKind::Metal,
        // Treat Unspecified the same as CPU so callers that leave the enum at
        // its proto3 default get a sensible CPU run rather than an error.
        ProtoDevice::Cpu | ProtoDevice::Unspecified => DeviceKind::Cpu,
    };

    let dtype = match ProtoDType::try_from(settings.dtype).unwrap_or(ProtoDType::DtypeUnspecified) {
        ProtoDType::F16 => DTypeKind::F16,
        ProtoDType::Bf16 => DTypeKind::Bf16,
        ProtoDType::F32 | ProtoDType::DtypeUnspecified => DTypeKind::F32,
    };

    let image_size_limit_bytes = settings
        .image_size_limit_bytes
        .unwrap_or(DEFAULT_IMAGE_SIZE_LIMIT_BYTES);

    let max_decoded_image_bytes = match settings.max_decoded_image_bytes {
        None => DecodedImageBytesLimit::Default,
        Some(0) => DecodedImageBytesLimit::Unlimited,
        Some(n) => DecodedImageBytesLimit::Limited(n),
    };

    let allowed_image_dirs = canonicalize_allowed_dirs(&settings.allowed_image_dirs)?;

    let tokenizer_model_id = settings
        .tokenizer_model_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map_or_else(|| model_id.clone(), str::to_owned);

    let gpu_device_index = settings.gpu_device_index.unwrap_or(0);

    Ok(ValidatedSettings {
        model_id,
        device,
        gpu_device_index,
        dtype,
        max_sequence_length,
        image_size_limit_bytes,
        max_decoded_image_bytes,
        allowed_image_dirs,
        tokenizer_model_id,
        chunking_config,
    })
}

fn resolve_chunking_config(
    cfg: Option<ProtoChunkingConfig>,
    max_sequence_length: u32,
) -> Result<ResolvedChunkingConfig> {
    let Some(cfg) = cfg else {
        // Default derivation: max_chunk_tokens = max_sequence_length,
        // min_chunk_tokens = 0.
        return Ok(ResolvedChunkingConfig {
            max_chunk_tokens: max_sequence_length,
            min_chunk_tokens: 0,
        });
    };

    let max_chunk_tokens = cfg.max_chunk_tokens.unwrap_or(max_sequence_length);
    let min_chunk_tokens = cfg.min_chunk_tokens.unwrap_or(0);

    if max_chunk_tokens == 0 {
        return Err(MmEmbeddingError::invalid_argument(
            "chunking_config.max_chunk_tokens must be greater than 0",
        ));
    }
    if max_chunk_tokens > max_sequence_length {
        return Err(MmEmbeddingError::invalid_argument(format!(
            "chunking_config.max_chunk_tokens ({max_chunk_tokens}) must not exceed max_sequence_length ({max_sequence_length})"
        )));
    }
    // Mirror command-utils::HierarchicalChunkingConfig::validate(): min < max
    // is a hard requirement; equality means the chunker has no room to merge
    // and is rejected.
    if min_chunk_tokens >= max_chunk_tokens {
        return Err(MmEmbeddingError::invalid_argument(format!(
            "chunking_config.min_chunk_tokens ({min_chunk_tokens}) must be less than max_chunk_tokens ({max_chunk_tokens})"
        )));
    }

    Ok(ResolvedChunkingConfig {
        max_chunk_tokens,
        min_chunk_tokens,
    })
}

fn canonicalize_allowed_dirs(raw: &[String]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::with_capacity(raw.len());
    for entry in raw {
        match std::fs::canonicalize(entry) {
            Ok(p) => out.push(p),
            Err(_) => {
                return Err(MmEmbeddingError::invalid_argument(format!(
                    "allowed_image_dirs entry not canonicalizable: {entry}"
                )));
            }
        }
    }
    Ok(out)
}
