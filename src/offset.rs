//! Convert tokenizer byte offsets into char (Unicode scalar value) indices.
//!
//! `tokenizers::Encoding::get_offsets()` returns `(byte_start, byte_end)`
//! pairs against the input string, but the result protocol exposes
//! char-indexed `begin_position` / `end_position`, so this module bridges
//! the two.
//!
//! Conversion strategy:
//! 1. Walk `text.char_indices()` once to build a sorted `Vec<usize>` mapping
//!    char-index → byte-offset and a sentinel for `text.len()`.
//! 2. Binary-search each tokenizer byte offset in that table. A match means
//!    the offset lands on a char boundary; otherwise it falls inside a
//!    multi-byte sequence and we treat that as a backend error rather than
//!    silently rounding (which would scramble the output position metadata).

use crate::error::{MmEmbeddingError, Result};

/// Per-char index → starting byte offset. The final entry equals `text.len()`
/// so a range `(byte_start, byte_end)` of `(0, text.len())` round-trips
/// cleanly.
#[derive(Debug, Clone)]
pub struct CharByteMap {
    starts: Vec<usize>,
}

impl CharByteMap {
    #[must_use]
    pub fn new(text: &str) -> Self {
        let mut starts = Vec::with_capacity(text.len() / 2 + 1);
        for (idx, _) in text.char_indices() {
            starts.push(idx);
        }
        starts.push(text.len()); // sentinel
        Self { starts }
    }

    /// Total number of chars (= `text.chars().count()`).
    #[must_use]
    pub fn char_len(&self) -> usize {
        // Subtract 1 for the sentinel.
        self.starts.len().saturating_sub(1)
    }

    /// Convert a byte offset into a char index, requiring char-boundary
    /// alignment.
    pub fn byte_to_char(&self, byte_offset: usize) -> Result<usize> {
        match self.starts.binary_search(&byte_offset) {
            Ok(char_idx) => Ok(char_idx),
            Err(_) => Err(MmEmbeddingError::backend(format!(
                "token byte offset ({byte_offset}) does not align with char boundary"
            ))),
        }
    }

    /// Convert a `(byte_start, byte_end)` pair into `(char_start, char_end)`.
    pub fn byte_range_to_char_range(
        &self,
        byte_start: usize,
        byte_end: usize,
    ) -> Result<(usize, usize)> {
        let cs = self.byte_to_char(byte_start)?;
        let ce = self.byte_to_char(byte_end)?;
        if cs > ce {
            return Err(MmEmbeddingError::backend(format!(
                "token byte range inverted: ({byte_start}, {byte_end})"
            )));
        }
        Ok((cs, ce))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_byte_equals_char() {
        let text = "hello";
        let map = CharByteMap::new(text);
        assert_eq!(map.char_len(), 5);
        assert_eq!(map.byte_range_to_char_range(0, 5).unwrap(), (0, 5));
        assert_eq!(map.byte_range_to_char_range(1, 4).unwrap(), (1, 4));
    }

    #[test]
    fn japanese_three_byte_chars() {
        let text = "あい"; // each char is 3 bytes in UTF-8
        let map = CharByteMap::new(text);
        assert_eq!(map.char_len(), 2);
        // (0, 6) bytes => (0, 2) chars
        assert_eq!(map.byte_range_to_char_range(0, 6).unwrap(), (0, 2));
        // (3, 6) bytes => (1, 2) chars
        assert_eq!(map.byte_range_to_char_range(3, 6).unwrap(), (1, 2));
    }

    #[test]
    fn emoji_four_byte_chars() {
        let text = "a🦀b"; // 1 + 4 + 1 = 6 bytes, 3 chars
        let map = CharByteMap::new(text);
        assert_eq!(map.char_len(), 3);
        assert_eq!(map.byte_range_to_char_range(1, 5).unwrap(), (1, 2));
    }

    #[test]
    fn byte_offset_inside_multibyte_char_errors() {
        let text = "あ"; // 3 bytes, 1 char
        let map = CharByteMap::new(text);
        let err = map.byte_to_char(1).expect_err("expected boundary error");
        assert!(err.to_string().starts_with("BackendError:"));
        assert!(err.to_string().contains("char boundary"));
    }

    #[test]
    fn inverted_range_errors() {
        let text = "hello";
        let map = CharByteMap::new(text);
        let err = map.byte_range_to_char_range(3, 1).expect_err("inverted");
        assert!(err.to_string().starts_with("BackendError:"));
    }

    #[test]
    fn empty_string_zero_length_range() {
        let map = CharByteMap::new("");
        assert_eq!(map.char_len(), 0);
        assert_eq!(map.byte_range_to_char_range(0, 0).unwrap(), (0, 0));
    }

    #[test]
    fn nfc_composed_form_one_char_per_accent() {
        // "é" in NFC is a single 2-byte char (U+00E9).
        let text = "Café";
        let map = CharByteMap::new(text);
        assert_eq!(map.char_len(), 4);
        // (0, 5) bytes => (0, 4) chars
        assert_eq!(map.byte_range_to_char_range(0, 5).unwrap(), (0, 4));
    }

    #[test]
    fn nfd_decomposed_form_each_codepoint_separate() {
        // "Café" in NFD is C + a + f + e + COMBINING ACUTE ACCENT = 5 chars.
        let text = "Cafe\u{301}";
        let map = CharByteMap::new(text);
        assert_eq!(map.char_len(), 5);
        // Combining accent is 2 bytes long, base 'e' is 1 byte.
        // 'C' at byte 0, 'a' at 1, 'f' at 2, 'e' at 3, accent starts at 4
        // and ends at 6. (3, 6) bytes => (3, 5) chars.
        assert_eq!(map.byte_range_to_char_range(3, 6).unwrap(), (3, 5));
    }
}
