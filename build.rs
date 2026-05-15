use std::io::Result;

fn main() -> Result<()> {
    // type_attribute(".", ...) applies serde + schemars derives to every
    // generated message so settings_schema! / arguments_schema! work without
    // hand-written wrappers, matching the pattern used by embedding-llm.
    prost_build::Config::new()
        .protoc_arg("--experimental_allow_proto3_optional")
        .type_attribute(
            ".",
            "#[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]",
        )
        .compile_protos(
            &[
                "protobuf/mm_embedding_settings.proto",
                "protobuf/embed_text_args.proto",
                "protobuf/embed_image_args.proto",
                "protobuf/mm_embedding_result.proto",
            ],
            &["protobuf"],
        )?;
    Ok(())
}
