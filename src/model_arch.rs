//! HF `config.json` -> model-architecture detection.
//!
//! `MmBackend::load` calls [`detect_architecture`] before constructing the
//! actual fastembed model, so we can route the load to the right
//! `from_hf` constructor. The detection rule is intentionally a plain
//! `architectures[0]` string match: this keeps the contract auditable
//! (a single grep tells you what is and is not supported) and the
//! pure-bytes parser [`parse_architecture`] makes the rule unit-testable
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchitectureKind {
    /// Qwen3-VL multimodal embedding (text + image share one space).
    Qwen3Vl,
    /// Qwen3 text-only causal-LM embedding (Qwen3-Embedding-* family).
    Qwen3Text,
}

/// Fetch `config.json` from `repo` (via HF Hub cache; honors `HF_HOME`)
/// and classify the model architecture.
pub fn detect_architecture(repo: &ApiRepo) -> Result<ArchitectureKind> {
    let cfg_path = repo
        .get("config.json")
        .map_err(|e| MmEmbeddingError::backend(format!("fetch config.json: {e}")))?;
    let bytes = std::fs::read(&cfg_path)?;
    parse_architecture(&bytes)
}

/// Pure-bytes counterpart of [`detect_architecture`]. Reads
/// `architectures[0]` from the supplied `config.json` payload and returns
/// the corresponding [`ArchitectureKind`], or an error variant the caller
/// can surface (BackendError for malformed JSON, UnsupportedModel for an
/// otherwise-valid config whose architecture we have no loader for).
pub(crate) fn parse_architecture(bytes: &[u8]) -> Result<ArchitectureKind> {
    let cfg: ConfigArchitectures = serde_json::from_slice(bytes)
        .map_err(|e| MmEmbeddingError::backend(format!("parse config.json: {e}")))?;
    let arch = cfg
        .architectures
        .first()
        .ok_or_else(|| MmEmbeddingError::backend("config.json missing architectures[0]"))?;
    match arch.as_str() {
        "Qwen3VLForConditionalGeneration" => Ok(ArchitectureKind::Qwen3Vl),
        "Qwen3ForCausalLM" => Ok(ArchitectureKind::Qwen3Text),
        other => Err(MmEmbeddingError::unsupported_model(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG_QWEN3_VL: &[u8] =
        br#"{"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl"}"#;
    const CONFIG_QWEN3_TEXT: &[u8] =
        br#"{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}"#;
    const CONFIG_MODERNBERT: &[u8] =
        br#"{"architectures": ["ModernBertModel"], "model_type": "modernbert"}"#;

    #[test]
    fn parse_architecture_qwen3_vl() {
        assert_eq!(
            parse_architecture(CONFIG_QWEN3_VL).unwrap(),
            ArchitectureKind::Qwen3Vl
        );
    }

    #[test]
    fn parse_architecture_qwen3_text() {
        assert_eq!(
            parse_architecture(CONFIG_QWEN3_TEXT).unwrap(),
            ArchitectureKind::Qwen3Text
        );
    }

    #[test]
    fn parse_architecture_unsupported_yields_unsupported_model() {
        let err = parse_architecture(CONFIG_MODERNBERT).unwrap_err();
        let s = err.to_string();
        assert!(s.starts_with("UnsupportedModel:"), "got: {s}");
        assert!(s.contains("ModernBertModel"), "got: {s}");
    }

    #[test]
    fn parse_architecture_missing_field_yields_backend_error() {
        let err = parse_architecture(b"{}").unwrap_err();
        let s = err.to_string();
        assert!(s.starts_with("BackendError:"), "got: {s}");
        assert!(s.contains("missing architectures[0]"), "got: {s}");
    }

    #[test]
    fn parse_architecture_malformed_json_yields_backend_error() {
        let err = parse_architecture(b"not json").unwrap_err();
        let s = err.to_string();
        assert!(s.starts_with("BackendError:"), "got: {s}");
        assert!(s.contains("parse config.json"), "got: {s}");
    }
}
