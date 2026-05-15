//! Tests for byte→char offset conversion. The inline tests in src/offset.rs
//! cover synthetic strings; this file uses the on-disk fixtures so the
//! conversion stays correct against real UTF-8 sequences we will see in
//! production (Japanese, emoji, NFD-decomposed accents).

use mm_embedding_runner::offset::CharByteMap;
use std::path::PathBuf;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture_text(name: &str) -> String {
    std::fs::read_to_string(PathBuf::from(FIXTURES).join(name))
        .unwrap_or_else(|_| panic!("missing fixture: {name}"))
}

#[test]
fn ascii_fixture_byte_count_equals_char_count() {
    let text = fixture_text("sample_text_ascii.txt");
    let map = CharByteMap::new(&text);
    assert_eq!(map.char_len(), text.chars().count());
    assert_eq!(map.char_len(), text.len()); // ASCII only ⇒ bytes == chars
}

#[test]
fn japanese_fixture_distinguishes_byte_and_char_counts() {
    let text = fixture_text("sample_text_ja.txt");
    let map = CharByteMap::new(&text);
    assert_eq!(map.char_len(), text.chars().count());
    // Byte length should exceed char count because Japanese chars are 3 bytes.
    assert!(
        map.char_len() < text.len(),
        "ja text expected to be multi-byte"
    );
}

#[test]
fn emoji_fixture_full_range_round_trip() {
    let text = fixture_text("sample_text_emoji.txt");
    let map = CharByteMap::new(&text);
    let (cs, ce) = map.byte_range_to_char_range(0, text.len()).unwrap();
    assert_eq!(cs, 0);
    assert_eq!(ce, text.chars().count());
}

#[test]
fn nfc_fixture_one_char_per_accent() {
    let text = fixture_text("sample_text_nfc.txt");
    let map = CharByteMap::new(&text);
    // Expect chars < bytes (each accented letter is 2 bytes).
    assert!(map.char_len() < text.len());
}

#[test]
fn nfd_fixture_more_chars_than_nfc_for_same_visible_text() {
    let nfc = fixture_text("sample_text_nfc.txt");
    let nfd = fixture_text("sample_text_nfd.txt");
    let nfc_map = CharByteMap::new(&nfc);
    let nfd_map = CharByteMap::new(&nfd);
    // NFD splits each accented letter into base + combining mark, so the
    // char count is larger for the same number of "visible" graphemes.
    assert!(
        nfd_map.char_len() > nfc_map.char_len(),
        "nfd ({}) should have more chars than nfc ({})",
        nfd_map.char_len(),
        nfc_map.char_len()
    );
}

#[test]
fn boundary_violation_in_japanese_fixture() {
    let text = fixture_text("sample_text_ja.txt");
    let map = CharByteMap::new(&text);
    // Pick a byte offset 1 past the start of the first char: in 3-byte UTF-8
    // this is guaranteed to be a non-boundary position.
    let err = map.byte_to_char(1).expect_err("expected boundary error");
    assert!(err.to_string().starts_with("BackendError:"));
}
