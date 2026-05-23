//! Locks the HuggingFace cache-location contract that both the runner's
//! tokenizer fetch and the (vendored-fork) `Qwen3VLEmbedding::from_hf` weights
//! fetch depend on: with `HF_HOME` set, the hf-hub cache resolves to
//! `$HF_HOME/hub`. This regression guard needs no network or GPU. See the
//! Cargo.toml `[patch.crates-io]` note for why the fork is needed.
//!
//! `HF_HOME` is process-global, so these tests run single-threaded (the CI
//! lane already passes `--test-threads=1`) and restore the prior value.

mod common;

use common::with_hf_home;
use hf_hub::Cache;

#[test]
fn hf_home_routes_cache_to_hub_subdir() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let resolved = with_hf_home(tmp.path(), || Cache::from_env().path().clone());
    assert_eq!(
        resolved,
        tmp.path().join("hub"),
        "HF_HOME must resolve the hf-hub cache to $HF_HOME/hub",
    );
}

#[test]
fn hf_home_cache_dir_is_distinct_from_default() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let resolved = with_hf_home(tmp.path(), || Cache::from_env().path().clone());
    let default_path = Cache::default().path().clone();
    assert_ne!(
        resolved, default_path,
        "with HF_HOME set, the cache must not fall back to the default \
         $HOME/.cache/huggingface/hub location",
    );
}
