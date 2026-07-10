use mm_embedding_runner::protobuf::mm_embedding::{
    ChunkingConfig, DType, Device, EmbedImageArgs, EmbedTextArgs, ImageItem, MmEmbeddingResult,
    ModelInfo, MultimodalEmbeddingRunnerSettings, OnnxPooling, image_item::Source as ImageSource,
    mm_embedding_result::Embedding,
};
use prost::Message;

#[test]
fn settings_roundtrip() {
    let original = MultimodalEmbeddingRunnerSettings {
        model_id: "Qwen/Qwen3-VL-Embedding-2B".to_string(),
        device: Device::Cuda as i32,
        gpu_device_index: Some(1),
        dtype: DType::F16 as i32,
        max_sequence_length: 2048,
        image_size_limit_bytes: Some(10 * 1024 * 1024),
        max_decoded_image_bytes: Some(0),
        allowed_image_dirs: vec!["/tmp/a".to_string(), "/tmp/b".to_string()],
        tokenizer_model_id: Some("custom/tokenizer".to_string()),
        chunking_config: Some(ChunkingConfig {
            max_chunk_tokens: Some(2048),
            min_chunk_tokens: Some(0),
        }),
        onnx_model_file: Some("onnx/model_int8.onnx".to_string()),
        onnx_pooling: OnnxPooling::Cls as i32,
    };
    let bytes = original.encode_to_vec();
    let decoded = MultimodalEmbeddingRunnerSettings::decode(&bytes[..]).unwrap();
    assert_eq!(original, decoded);
}

#[test]
fn embed_text_args_roundtrip() {
    let original = EmbedTextArgs {
        text: "hello world".to_string(),
    };
    let bytes = original.encode_to_vec();
    let decoded = EmbedTextArgs::decode(&bytes[..]).unwrap();
    assert_eq!(original, decoded);
}

#[test]
fn embed_image_args_roundtrip_all_source_variants() {
    let items = vec![
        ImageItem {
            label: Some("file-based".to_string()),
            source: Some(ImageSource::FilePath("/tmp/x.png".to_string())),
        },
        ImageItem {
            label: None,
            source: Some(ImageSource::Data(b"\x89PNG\r\n".to_vec())),
        },
        ImageItem {
            label: Some("url-based".to_string()),
            // url variant is reserved on the wire but rejected at runtime;
            // round-trip should still preserve it.
            source: Some(ImageSource::Url("https://example.com/x.png".to_string())),
        },
    ];
    let original = EmbedImageArgs { items };
    let bytes = original.encode_to_vec();
    let decoded = EmbedImageArgs::decode(&bytes[..]).unwrap();
    assert_eq!(original, decoded);
}

#[test]
fn result_roundtrip_matches_field_names_with_embedding_llm() {
    // The set of fields here MUST match embedding-llm's EmbeddingLlmResult
    // wire layout so JSON consumers can decode either runner's payload
    // interchangeably.
    let original = MmEmbeddingResult {
        embeddings: vec![
            Embedding {
                values: vec![0.1, 0.2, 0.3],
                begin_position: 0,
                end_position: 10,
                content: "hello".to_string(),
            },
            Embedding {
                values: vec![0.4, 0.5, 0.6],
                begin_position: 10,
                end_position: 20,
                content: "world".to_string(),
            },
        ],
        model_info: Some(ModelInfo {
            model_name: "Qwen/Qwen3-VL-Embedding-2B".to_string(),
            embedding_dimension: 2048,
            dtype_used: "F32".to_string(),
            supports_vision: true,
            supports_audio: false,
            audio_sample_rate: None,
        }),
    };
    let bytes = original.encode_to_vec();
    let decoded = MmEmbeddingResult::decode(&bytes[..]).unwrap();
    assert_eq!(original, decoded);
}

#[test]
fn result_json_field_names_are_canonical() {
    // Snake_case is the wire-level canonical form; the Serialize impl emitted
    // by prost/serde should preserve that so JSON shape stays aligned with
    // embedding-llm consumers.
    let result = MmEmbeddingResult {
        embeddings: vec![Embedding {
            values: vec![1.0],
            begin_position: 0,
            end_position: 5,
            content: "hello".to_string(),
        }],
        model_info: None,
    };
    let json = serde_json::to_string(&result).unwrap();
    for expected in [
        "\"embeddings\"",
        "\"values\"",
        "\"begin_position\"",
        "\"end_position\"",
        "\"content\"",
    ] {
        assert!(
            json.contains(expected),
            "JSON should contain {expected}, but got: {json}"
        );
    }
}
