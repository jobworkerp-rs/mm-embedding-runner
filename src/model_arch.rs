//! HF `config.json` -> model-architecture detection.
//!
//! `MmBackend::load` calls [`detect_model_config`] before constructing the
//! actual fastembed model, so we can route the load to the right
//! `from_hf` constructor. The detection rule is intentionally a plain
//! `architectures[0]` string match: this keeps the contract auditable
//! (a single grep tells you what is and is not supported) and the
//! pure-bytes parser [`parse_model_config`] makes the rule unit-testable
//! offline.

use hf_hub::api::sync::ApiRepo;
use serde::Deserialize;

use crate::error::{MmEmbeddingError, Result};

/// Minimal projection of `config.json` — we only need the architecture
/// tag; serde ignores the rest of the file.
#[derive(Deserialize)]
struct ConfigArchitectures {
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureKind {
    /// Qwen3-VL multimodal embedding (text + image share one space).
    Qwen3Vl,
    /// Qwen3 text-only causal-LM embedding (Qwen3-Embedding-* family).
    Qwen3Text,
    /// ModernBERT text embedding exported as ONNX.
    ModernBertOnnx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelConfigMetadata {
    pub architecture: ArchitectureKind,
    pub hidden_size: Option<usize>,
}

/// Fetch a file from `repo` (via HF Hub cache; honors `HF_HOME`) and read
/// its bytes into memory. Shared HF-fetch primitive used by config-json
/// detection here and by the backend's ONNX/tokenizer file loading.
pub(crate) fn read_hf_file(repo: &ApiRepo, path: &str) -> Result<Vec<u8>> {
    let file = repo
        .get(path)
        .map_err(|e| MmEmbeddingError::backend(format!("fetch {path}: {e}")))?;
    std::fs::read(file).map_err(Into::into)
}

/// Fetch and parse the parts of `config.json` required by backend loading:
/// `architectures[0]` (routes to the right loader) and `hidden_size`
/// (embedding dimension for ONNX backends).
pub fn detect_model_config(repo: &ApiRepo) -> Result<ModelConfigMetadata> {
    parse_model_config(&read_hf_file(repo, "config.json")?)
}

/// Pure-bytes parser for `config.json`. Reads `architectures[0]` and
/// `hidden_size` from the supplied payload and returns the corresponding
/// [`ModelConfigMetadata`], or an error variant the caller can surface
/// (BackendError for malformed JSON, UnsupportedModel for an
/// otherwise-valid config whose architecture we have no loader for).
pub(crate) fn parse_model_config(bytes: &[u8]) -> Result<ModelConfigMetadata> {
    let cfg: ConfigArchitectures = serde_json::from_slice(bytes)
        .map_err(|e| MmEmbeddingError::backend(format!("parse config.json: {e}")))?;
    let arch = cfg
        .architectures
        .first()
        .ok_or_else(|| MmEmbeddingError::backend("config.json missing architectures[0]"))?;
    let architecture = match arch.as_str() {
        "Qwen3VLForConditionalGeneration" => Ok(ArchitectureKind::Qwen3Vl),
        "Qwen3ForCausalLM" => Ok(ArchitectureKind::Qwen3Text),
        "ModernBertModel" => Ok(ArchitectureKind::ModernBertOnnx),
        other => Err(MmEmbeddingError::unsupported_model(other)),
    }?;
    Ok(ModelConfigMetadata {
        architecture,
        hidden_size: cfg.hidden_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG_QWEN3_VL: &[u8] =
        br#"{"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl"}"#;
    const CONFIG_QWEN3_TEXT: &[u8] =
        br#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}"#;
    const CONFIG_MODERNBERT: &[u8] =
        br#"{"architectures": ["ModernBertModel"], "model_type": "modernbert", "hidden_size": 768}"#;

    #[test]
    fn parse_architecture_qwen3_vl() {
        assert_eq!(
            parse_model_config(CONFIG_QWEN3_VL).unwrap().architecture,
            ArchitectureKind::Qwen3Vl
        );
    }

    #[test]
    fn parse_architecture_qwen3_text() {
        assert_eq!(
            parse_model_config(CONFIG_QWEN3_TEXT).unwrap().architecture,
            ArchitectureKind::Qwen3Text
        );
    }

    #[test]
    fn parse_architecture_modernbert_onnx() {
        assert_eq!(
            parse_model_config(CONFIG_MODERNBERT).unwrap().architecture,
            ArchitectureKind::ModernBertOnnx
        );
    }

    #[test]
    fn parse_model_config_modernbert_reads_hidden_size() {
        let cfg = parse_model_config(CONFIG_MODERNBERT).unwrap();
        assert_eq!(cfg.architecture, ArchitectureKind::ModernBertOnnx);
        assert_eq!(cfg.hidden_size, Some(768));
    }

    #[test]
    fn parse_architecture_unknown_yields_unsupported_model() {
        let err =
            parse_model_config(br#"{"architectures": ["SomethingElse"], "model_type": "unknown"}"#)
                .unwrap_err();
        let s = err.to_string();
        assert!(s.starts_with("UnsupportedModel:"), "got: {s}");
        assert!(s.contains("SomethingElse"), "got: {s}");
    }

    #[test]
    fn parse_architecture_missing_field_yields_backend_error() {
        let err = parse_model_config(b"{}").unwrap_err();
        let s = err.to_string();
        assert!(s.starts_with("BackendError:"), "got: {s}");
        assert!(s.contains("missing architectures[0]"), "got: {s}");
    }

    #[test]
    fn parse_architecture_malformed_json_yields_backend_error() {
        let err = parse_model_config(b"not json").unwrap_err();
        let s = err.to_string();
        assert!(s.starts_with("BackendError:"), "got: {s}");
        assert!(s.contains("parse config.json"), "got: {s}");
    }
}
