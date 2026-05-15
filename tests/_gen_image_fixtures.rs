//! Run with: `cargo test --test _gen_image_fixtures -- --ignored generate -- --nocapture`
//! Regenerates the image fixtures committed under tests/fixtures/. Keep
//! ignored so it doesn't run as part of the normal test suite.

use image::{Rgb, RgbImage};
use std::fs;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

#[test]
#[ignore]
fn generate() {
    let dir = fixtures_dir();
    fs::create_dir_all(&dir).unwrap();

    // tiny_valid.png: 8x8 solid magenta. Smaller than typical model patch so
    // it remains a meaningful "valid header + tiny content" sample.
    let mut tiny = RgbImage::new(8, 8);
    for px in tiny.pixels_mut() {
        *px = Rgb([255, 0, 255]);
    }
    tiny.save(dir.join("tiny_valid.png")).unwrap();

    // tiny_valid.jpg: same data but JPEG-encoded so format detection covers
    // both PNG and JPEG entry points.
    tiny.save(dir.join("tiny_valid.jpg")).unwrap();

    // invalid_header.bin: arbitrary non-image bytes. The first bytes happen
    // to spell "NOTAPNG" so test failures are obvious when inspected.
    fs::write(dir.join("invalid_header.bin"), b"NOTAPNG_arbitrary_bytes").unwrap();

    // zip_bomb.png: 10000x10000 single-color PNG. RGBA decode would be
    // ~400 MB, well above the 256 MiB default cap, but the on-disk size
    // stays in the tens of kilobytes thanks to PNG run-length encoding of a
    // uniform image.
    let huge = RgbImage::from_pixel(10_000, 10_000, Rgb([0, 0, 0]));
    huge.save(dir.join("zip_bomb.png")).unwrap();

    // Sentinel keepers for the allowed-dir canonicalization tests live in
    // dedicated subdirectories. Create them empty if missing.
    for sub in ["allowed_dir_a", "allowed_dir_b"] {
        let subdir = dir.join(sub);
        fs::create_dir_all(&subdir).unwrap();
        let keep = subdir.join(".keep");
        if !keep.exists() {
            fs::write(&keep, b"").unwrap();
        }
    }
}
