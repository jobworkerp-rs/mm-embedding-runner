# Test fixtures

ファイルは `cargo test` から相対パス参照される。各種類の趣旨は以下のとおり。

## 画像系（P2-2 で使用）
- `tiny_valid.png` — 8x8 程度の正常 PNG。基本ケース通過用
- `tiny_valid.jpg` — JPEG ヘッダ確認用
- `invalid_header.bin` — 画像ヘッダの無いバイナリ。`InvalidArgument: invalid image header` ガード用
- `zip_bomb.png` — 圧縮後は小さいが decode 後 bitmap が巨大になる PNG。`InvalidArgument: decoded size exceeds max_decoded_image_bytes` ガード用

これらは以下のヘルパー（`#[ignore]` 付き dev 専用テスト関数）で再生成可能。Git に直接 commit する。

```bash
cargo test --test _gen_image_fixtures -- --ignored generate
cargo test --test _gen_text_fixtures  -- --ignored generate
```

## テキスト系（P3 で使用）
- `sample_text_*.txt` — chunking ロジックの char offset テスト用
- `tokenizer.json` — Qwen3-VL-Embedding-2B 配下からコピー（Apache 2.0）

## 許可ディレクトリ系（P2-1 で使用）
- `allowed_dir_a/.keep` / `allowed_dir_b/.keep` — canonical 化検証用ダミー
