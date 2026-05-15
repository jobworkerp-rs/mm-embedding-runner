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

use anyhow::{Result, anyhow};
use jobworkerp_client::{plugins::MultiMethodPluginRunner, schema_to_json_string};
use prost::Message;
use std::collections::HashMap;

use crate::backend::MmBackend;
use crate::settings::{ValidatedSettings, validate};

pub const RUNNER_NAME: &str = "MultimodalEmbeddingRunner";
pub const METHOD_EMBED_TEXT: &str = "embed_text";
pub const METHOD_EMBED_IMAGE: &str = "embed_image";

/// Multimodal embedding plugin: text and image inputs share a single
/// Qwen3-VL-Embedding instance to guarantee vector-space alignment by
/// construction (specification.md §2.2 / §4.1).
pub struct MmEmbeddingPlugin {
    // None until load() succeeds. Holds the fastembed backend, the tokenizer
    // used for chunking accounting, and the resolved settings.
    backend: Option<MmBackend>,
}

impl MmEmbeddingPlugin {
    pub fn new() -> Self {
        Self { backend: None }
    }

    pub fn settings(&self) -> Option<&ValidatedSettings> {
        self.backend.as_ref().map(MmBackend::settings)
    }

    fn run_embed_text(&mut self, args: &[u8]) -> Result<Vec<u8>> {
        // Decode args first so that NotLoaded vs InvalidArgument is reported
        // for the right reason: decode failure is the caller's fault and is
        // surfaced regardless of whether the plugin has been loaded yet.
        let decoded = protobuf::mm_embedding::EmbedTextArgs::decode(args)
            .map_err(|e| anyhow!("InvalidArgument: failed to decode EmbedTextArgs: {e}"))?;
        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| anyhow!("NotLoaded: model is not loaded"))?;
        embed_text_impl(backend, decoded)
    }

    fn run_embed_image(&mut self, args: &[u8]) -> Result<Vec<u8>> {
        let decoded = protobuf::mm_embedding::EmbedImageArgs::decode(args)
            .map_err(|e| anyhow!("InvalidArgument: failed to decode EmbedImageArgs: {e}"))?;
        let backend = self
            .backend
            .as_ref()
            .ok_or_else(|| anyhow!("NotLoaded: model is not loaded"))?;
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
        let proto =
            protobuf::mm_embedding::MultimodalEmbeddingRunnerSettings::decode(&settings[..])
                .map_err(|e| anyhow!("InvalidArgument: failed to decode settings: {e}"))?;
        let validated = validate(proto).map_err(|e| anyhow!(e.to_string()))?;
        let backend = MmBackend::load(validated).map_err(|e| anyhow!(e.to_string()))?;
        self.backend = Some(backend);
        Ok(())
    }

    fn run(
        &mut self,
        args: Vec<u8>,
        metadata: HashMap<String, String>,
        using: Option<&str>,
    ) -> (Result<Vec<u8>>, HashMap<String, String>) {
        let result = match using {
            None => Err(anyhow!("InvalidArgument: method not specified")),
            Some(METHOD_EMBED_TEXT) => self.run_embed_text(&args),
            Some(METHOD_EMBED_IMAGE) => self.run_embed_image(&args),
            Some(other) => Err(anyhow!(
                "InvalidArgument: unknown method '{other}' (supported: embed_text, embed_image)"
            )),
        };
        (result, metadata)
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

fn embed_text_impl(
    backend: &MmBackend,
    args: protobuf::mm_embedding::EmbedTextArgs,
) -> Result<Vec<u8>> {
    if args.text.trim().is_empty() {
        return Err(anyhow!("InvalidArgument: text must not be empty"));
    }

    let chunks = chunking::chunk_text(
        &args.text,
        backend.tokenizer(),
        backend.settings().chunking_config,
    )
    .map_err(|e| anyhow!(e.to_string()))?;

    // Batch the chunk contents into a single embed_texts call so we hit the
    // backend once regardless of how many chunks the text yielded.
    let chunk_texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let vectors = backend
        .embed_texts(&chunk_texts)
        .map_err(|e| anyhow!(e.to_string()))?;

    if vectors.len() != chunks.len() {
        return Err(anyhow!(
            "BackendError: embed_texts returned {} vectors for {} chunks",
            vectors.len(),
            chunks.len()
        ));
    }

    let dim = vectors.first().map(Vec::len).unwrap_or(0);
    let embeddings = chunks
        .into_iter()
        .zip(vectors)
        .map(
            |(c, v)| protobuf::mm_embedding::mm_embedding_result::Embedding {
                values: v,
                begin_position: c.char_start as u32,
                end_position: c.char_end as u32,
                content: c.content,
            },
        )
        .collect();

    let result = protobuf::mm_embedding::MmEmbeddingResult {
        embeddings,
        model_info: Some(build_model_info(backend, dim)),
    };
    Ok(result.encode_to_vec())
}

fn embed_image_impl(
    backend: &MmBackend,
    args: protobuf::mm_embedding::EmbedImageArgs,
) -> Result<Vec<u8>> {
    let prepared = image_validation::prepare_images(&args.items, backend.settings())
        .map_err(|e| anyhow!(e.to_string()))?;

    let byte_refs: Vec<&[u8]> = prepared.iter().map(|p| p.bytes.as_slice()).collect();
    let vectors = backend
        .embed_image_bytes(&byte_refs)
        .map_err(|e| anyhow!(e.to_string()))?;

    if vectors.len() != prepared.len() {
        return Err(anyhow!(
            "BackendError: embed_image_bytes returned {} vectors for {} inputs",
            vectors.len(),
            prepared.len()
        ));
    }

    let dim = vectors.first().map(Vec::len).unwrap_or(0);
    let embeddings = prepared
        .into_iter()
        .zip(vectors)
        .enumerate()
        .map(
            |(i, (p, v))| protobuf::mm_embedding::mm_embedding_result::Embedding {
                values: v,
                begin_position: i as u32,
                end_position: (i as u32) + 1,
                content: p.source_hint,
            },
        )
        .collect();

    let result = protobuf::mm_embedding::MmEmbeddingResult {
        embeddings,
        model_info: Some(build_model_info(backend, dim)),
    };
    Ok(result.encode_to_vec())
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

// C entry points used by the jobworkerp plugin loader. Both functions are
// crossed by panic::catch_unwind so a panicking constructor on a non-CUDA
// host (e.g. environment lookup failure) does not unwind across the FFI
// boundary into the host process. See specification.md §4.4.
#[unsafe(no_mangle)]
pub extern "C" fn load_multi_method_plugin() -> Box<dyn MultiMethodPluginRunner + Send + Sync> {
    std::panic::catch_unwind(|| {
        dotenvy::dotenv().ok();
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
