# Test fixtures

Files are referenced from `cargo test` by relative path. Each fixture group is described below.

## Image Fixtures (used by P2-2)

- `tiny_valid.png` - A valid PNG around 8x8 pixels, used for basic success cases
- `tiny_valid.jpg` - Used to verify JPEG header handling
- `invalid_header.bin` - Binary data without an image header, used for the `InvalidArgument: invalid image header` guard
- `zip_bomb.png` - A PNG that is small when compressed but expands to a large decoded bitmap, used for the `InvalidArgument: decoded size exceeds max_decoded_image_bytes` guard

These fixtures can be regenerated with the following helpers, which are dev-only ignored test functions. Commit the generated files directly to Git.

```bash
cargo test --test _gen_image_fixtures -- --ignored generate
cargo test --test _gen_text_fixtures  -- --ignored generate
```

## Text Fixtures (used by P3)

- `sample_text_*.txt` - Used by char-offset tests for chunking logic
- `tokenizer.json` - Copied from Qwen3-VL-Embedding-2B (Apache 2.0)

## Allowed Directory Fixtures (used by P2-1)

- `allowed_dir_a/.keep` / `allowed_dir_b/.keep` - Dummy files for canonicalization checks
