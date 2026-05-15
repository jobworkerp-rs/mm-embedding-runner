#![allow(improper_ctypes_definitions)]

pub mod backend;
pub mod chunking;
pub mod error;
pub mod image_validation;
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

use anyhow::Result;
use jobworkerp_client::{plugins::MultiMethodPluginRunner, schema_to_json_string};
use prost::Message;
use std::collections::HashMap;

use crate::backend::MmBackend;
use crate::error::{MmEmbeddingError, Result as MmResult};
use crate::settings::{ValidatedSettings, validate};

pub const RUNNER_NAME: &str = "MultimodalEmbeddingRunner";
pub const METHOD_EMBED_TEXT: &str = "embed_text";
pub const METHOD_EMBED_IMAGE: &str = "embed_image";

/// Multimodal embedding plugin. Text and image inputs are embedded by a
/// single shared Qwen3-VL-Embedding instance, so their vectors live in the
/// same space and are directly comparable (e.g. cosine similarity).
pub struct MmEmbeddingPlugin {
    backend: Option<MmBackend>,
}

impl MmEmbeddingPlugin {
    pub fn new() -> Self {
        Self { backend: None }
    }

    pub fn settings(&self) -> Option<&ValidatedSettings> {
        self.backend.as_ref().map(MmBackend::settings)
    }

    fn run_embed_text(&mut self, args: &[u8]) -> MmResult<Vec<u8>> {
        // Decode before the load check so a malformed payload is reported as
        // the caller's fault (InvalidArgument) regardless of load state.
        let decoded = protobuf::mm_embedding::EmbedTextArgs::decode(args)?;
        let backend = self.backend.as_ref().ok_or(MmEmbeddingError::NotLoaded)?;
        embed_text_impl(backend, decoded)
    }

    fn run_embed_image(&mut self, args: &[u8]) -> MmResult<Vec<u8>> {
        let decoded = protobuf::mm_embedding::EmbedImageArgs::decode(args)?;
        let backend = self.backend.as_ref().ok_or(MmEmbeddingError::NotLoaded)?;
        embed_image_impl(backend, decoded)
    }
}

impl Default for MmEmbeddingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiMethodPluginRunner for MmEmbeddingPlugin {
    fn name(&self) -> String {
        String::from(RUNNER_NAME)
    }

    fn description(&self) -> String {
        String::from(
            "Generate multimodal embeddings (text and image) with positional information in a shared vector space",
        )
    }

    fn load(&mut self, settings: Vec<u8>) -> Result<()> {
        let load = || -> MmResult<MmBackend> {
            let proto =
                protobuf::mm_embedding::MultimodalEmbeddingRunnerSettings::decode(&settings[..])?;
            MmBackend::load(validate(proto)?)
        };
        // anyhow only at the trait boundary; MmEmbeddingError owns the
        // prefixed Display, so error classification stays in error.rs.
        self.backend = Some(load().map_err(anyhow::Error::from)?);
        Ok(())
    }

    fn run(
        &mut self,
        args: Vec<u8>,
        metadata: HashMap<String, String>,
        using: Option<&str>,
    ) -> (Result<Vec<u8>>, HashMap<String, String>) {
        let result: MmResult<Vec<u8>> = match using {
            None => Err(MmEmbeddingError::invalid_argument("method not specified")),
            Some(METHOD_EMBED_TEXT) => self.run_embed_text(&args),
            Some(METHOD_EMBED_IMAGE) => self.run_embed_image(&args),
            Some(other) => Err(MmEmbeddingError::invalid_argument(format!(
                "unknown method '{other}' (supported: embed_text, embed_image)"
            ))),
        };
        (result.map_err(anyhow::Error::from), metadata)
    }

    fn cancel(&mut self) -> bool {
        // v0.1 is serial-only; cancel is a no-op so that callers can rely on
        // the return value to detect lack of cooperative cancellation.
        false
    }

    fn is_canceled(&self) -> bool {
        false
    }

    fn runner_settings_proto(&self) -> String {
        include_str!("../protobuf/mm_embedding_settings.proto").to_string()
    }

    fn method_proto_map(
        &self,
    ) -> HashMap<String, jobworkerp_client::jobworkerp::data::MethodSchema> {
        static CACHED: std::sync::OnceLock<
            HashMap<String, jobworkerp_client::jobworkerp::data::MethodSchema>,
        > = std::sync::OnceLock::new();
        CACHED
            .get_or_init(|| {
                let mut schemas = HashMap::new();
                let result_proto =
                    include_str!("../protobuf/mm_embedding_result.proto").to_string();
                schemas.insert(
                    METHOD_EMBED_TEXT.to_string(),
                    jobworkerp_client::jobworkerp::data::MethodSchema {
                        args_proto: include_str!("../protobuf/embed_text_args.proto").to_string(),
                        result_proto: result_proto.clone(),
                        description: Some(
                            "Embed text input. Long text is chunked into sliding windows; each window becomes one embedding entry.".to_string(),
                        ),
                        output_type:
                            jobworkerp_client::jobworkerp::data::StreamingOutputType::NonStreaming
                                as i32,
                        require_client_stream: false,
                        client_stream_data_proto: None,
                    },
                );
                schemas.insert(
                    METHOD_EMBED_IMAGE.to_string(),
                    jobworkerp_client::jobworkerp::data::MethodSchema {
                        args_proto: include_str!("../protobuf/embed_image_args.proto").to_string(),
                        result_proto,
                        description: Some(
                            "Embed image inputs into the shared multimodal vector space. Returns one embedding per input image in submission order.".to_string(),
                        ),
                        output_type:
                            jobworkerp_client::jobworkerp::data::StreamingOutputType::NonStreaming
                                as i32,
                        require_client_stream: false,
                        client_stream_data_proto: None,
                    },
                );
                schemas
            })
            .clone()
    }

    fn settings_schema(&self) -> String {
        schema_to_json_string!(
            protobuf::mm_embedding::MultimodalEmbeddingRunnerSettings,
            "settings_schema"
        )
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
    let dim = vectors.first().map(Vec::len).unwrap_or(0);
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
        model_info: Some(build_model_info(backend, dim)),
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

fn build_model_info(
    backend: &MmBackend,
    embedding_dimension: usize,
) -> protobuf::mm_embedding::ModelInfo {
    let dtype_used = match backend.settings().dtype {
        settings::DTypeKind::F32 => "F32",
        settings::DTypeKind::F16 => "F16",
        settings::DTypeKind::Bf16 => "BF16",
    };
    protobuf::mm_embedding::ModelInfo {
        model_name: backend.settings().model_id.clone(),
        embedding_dimension: embedding_dimension as u32,
        dtype_used: dtype_used.to_string(),
        supports_vision: true,
        supports_audio: false,
        audio_sample_rate: None,
    }
}

// catch_unwind keeps a panicking constructor (e.g. env lookup failure on a
// non-CUDA host) from unwinding across the FFI boundary into the host.
#[unsafe(no_mangle)]
pub extern "C" fn load_multi_method_plugin() -> Box<dyn MultiMethodPluginRunner + Send + Sync> {
    std::panic::catch_unwind(|| {
        dotenvy::dotenv().ok();
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            command_utils::util::tracing::tracing_init_from_env()
                .await
                .unwrap_or_default();
        });
        Box::new(MmEmbeddingPlugin::new()) as Box<dyn MultiMethodPluginRunner + Send + Sync>
    })
    .unwrap_or_else(|e| {
        tracing::error!(
            "load_multi_method_plugin panic: {:?}, returning default plugin",
            e
        );
        Box::new(MmEmbeddingPlugin::new())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn free_multi_method_plugin(ptr: Box<dyn MultiMethodPluginRunner + Send + Sync>) {
    drop(ptr);
}
