//! Cross-modal alignment check. The spec asks for a real cat image, but
//! sourcing one with a redistributable license is out of scope for the
//! integration suite. We use the synthetic `tiny_valid.png` fixture and
//! settle for the looser invariant that the two modalities produce vectors
//! of matching dimension and well-defined cosine similarity, while we wait
//! for a real fixture to be added later.

mod common;

use common::{FIXTURES, cosine, load_test_plugin};
use jobworkerp_client::plugins::MultiMethodPluginRunner;
use mm_embedding_runner::protobuf::mm_embedding::{
    EmbedImageArgs, EmbedTextArgs, ImageItem, MmEmbeddingResult, image_item::Source as ImageSource,
};
use mm_embedding_runner::{METHOD_EMBED_IMAGE, METHOD_EMBED_TEXT, MmEmbeddingPlugin};
use prost::Message;
use std::collections::HashMap;
use std::path::PathBuf;

fn embed_text(plugin: &mut MmEmbeddingPlugin, text: &str) -> Vec<f32> {
    let args = EmbedTextArgs {
        text: text.to_string(),
    }
    .encode_to_vec();
    let (result, _) = plugin.run(args, HashMap::new(), Some(METHOD_EMBED_TEXT));
    let bytes = result.expect("embed_text");
    let parsed = MmEmbeddingResult::decode(&bytes[..]).unwrap();
    parsed.embeddings[0].values.clone()
}

fn embed_image(plugin: &mut MmEmbeddingPlugin, bytes: Vec<u8>) -> Vec<f32> {
    let args = EmbedImageArgs {
        items: vec![ImageItem {
            source: Some(ImageSource::Data(bytes)),
            label: None,
        }],
    }
    .encode_to_vec();
    let (result, _) = plugin.run(args, HashMap::new(), Some(METHOD_EMBED_IMAGE));
    let bytes = result.expect("embed_image");
    let parsed = MmEmbeddingResult::decode(&bytes[..]).unwrap();
    parsed.embeddings[0].values.clone()
}

#[test]
#[ignore]
fn image_and_text_share_vector_space() {
    let mut plugin = load_test_plugin(vec![]);

    let img_bytes = std::fs::read(PathBuf::from(FIXTURES).join("tiny_valid.png")).unwrap();
    let img_vec = embed_image(&mut plugin, img_bytes);
    let text_vec_a = embed_text(
        &mut plugin,
        "Instruct: Retrieve relevant content.\nQuery: a magenta square",
    );
    let text_vec_b = embed_text(
        &mut plugin,
        "Instruct: Retrieve relevant content.\nQuery: a steam locomotive",
    );

    // Dimensions must match (single shared vector space).
    assert_eq!(img_vec.len(), text_vec_a.len());
    assert_eq!(img_vec.len(), text_vec_b.len());

    let sim_a = cosine(&img_vec, &text_vec_a);
    let sim_b = cosine(&img_vec, &text_vec_b);

    // Sanity: cosine sims must be finite and inside [-1, 1].
    assert!(sim_a.is_finite() && (-1.0..=1.0).contains(&sim_a));
    assert!(sim_b.is_finite() && (-1.0..=1.0).contains(&sim_b));

    // Loose check: the two queries should not produce identical similarity
    // (otherwise the model isn't doing anything useful with text content).
    assert!(
        (sim_a - sim_b).abs() > 1e-4,
        "expected different cosine similarities for different queries; sim_a={sim_a}, sim_b={sim_b}",
    );
}
