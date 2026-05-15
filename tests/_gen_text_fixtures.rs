//! Run with: `cargo test --test _gen_text_fixtures -- --ignored generate -- --nocapture`
//! Regenerates the text fixtures that need precise byte-level control (NFD
//! decomposition, long content). NFC fixtures can be hand-edited.

use std::fs;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
#[ignore]
fn generate() {
    let dir = fixtures_dir();

    // NFD form using explicit Unicode escapes so the bytes-on-disk are
    // unambiguous regardless of editor settings.
    // Base letter + combining mark:
    //   e + COMBINING ACUTE ACCENT (U+0301)
    //   a + COMBINING DIAERESIS (U+0308)
    let nfd = "Cafe\u{301} re\u{301}sume\u{301} nai\u{308}ve — NFD decomposed form: combining marks are separate code points after the base letter.";
    fs::write(dir.join("sample_text_nfd.txt"), nfd).unwrap();

    // Long sample for chunking trigger. Repeat the ja paragraph enough times
    // to exceed 4096 tokens at typical sub-word rates (~2x token/char ratio
    // for Qwen on CJK). 200 KB of UTF-8 is more than enough.
    let para = std::fs::read_to_string(dir.join("sample_text_ja.txt")).unwrap();
    let long: String = std::iter::repeat_n(para.as_str(), 200)
        .collect::<Vec<&str>>()
        .join("\n\n");
    fs::write(dir.join("sample_text_long.txt"), long).unwrap();
}
