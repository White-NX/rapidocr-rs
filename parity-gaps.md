# rapidocr-rs Parity Gaps

This file records known differences between the current Rust MVP and Python RapidOCR. These are not ignored forever; they are kept out of strict e2e gates until the expected behavior is understood and the implementation is closer to Python parity.

## Current Strong Gates

The current e2e parity fixtures cover:

- `ch_en_num.jpg`
- `text_det.jpg`
- `ch_en_num.jpg` and `text_det.jpg` as detection-only geometry checks
- `check_return_word_len.jpeg` as a dense-text detection-only geometry check
- `arabic.png`, `cyrillic.png`, `devanagari.jpg`, `japan.jpg`, and `korean.jpg` as cross-language detection-only geometry checks
- `en.jpg`
- `empty_black.jpg`
- `short.png`
- `black_font_color_transparent.png`
- `img_exif_orientation.jpg`
- `ch_doc_server.png` with cls enabled and detection-only
- `test_letterbox_like.jpg`
- `test_without_det.jpg`
- `text_vertical_words.png`
- `latin.jpg`
- `return_word_debug.jpg` with cls enabled
- `en_rec.jpg` as a recognition-only cls/no-cls long English line check
- `text_rec.jpg` as a recognition-only cls/no-cls normal-crop check
- `text_cls.jpg` as a recognition-only cls/no-cls 180-degree crop check
- `text_cls.jpg` as a Rust cls/no-cls golden

The current DBPostProcess parity fixtures additionally cover `black_font_color_transparent.png`, `return_word_debug.jpg`, `short.png`, `test_without_det.jpg`, and `ch_doc_server.png`.

Current representative metrics:

- `ch_en_num.jpg` and `text_det.jpg`: 18/18 lines matched, character accuracy about 0.976, mean center drift about 1.23 px.
- `ch_en_num.jpg` and `text_det.jpg` detection-only: 18/18 boxes matched, mean center drift about 1.23 px, mean corner drift about 1.45 px.
- `check_return_word_len.jpeg` detection-only: 28/28 boxes matched, mean center drift about 0.58 px, mean corner drift about 0.92 px.
- Cross-language detection-only: `arabic.png` 2/2, `cyrillic.png` 4/4, `devanagari.jpg` 4/4, `japan.jpg` 7/7, and `korean.jpg` 6/6 boxes matched with mean center drift below 0.70 px.
- `en.jpg`: 5/5 lines matched, exact text match, mean center drift about 0.21 px.
- `empty_black.jpg`: 0/0 lines matched.
- `short.png`: 0/0 lines matched.
- `black_font_color_transparent.png`: 3/3 lines matched, exact text match, mean center drift about 1.03 px.
- `img_exif_orientation.jpg`: 1/1 line matched, exact text match, mean center drift about 0.45 px.
- `ch_doc_server.png` with cls enabled: 2/2 lines matched, exact text match, mean corner drift about 0.45 px.
- `ch_doc_server.png` detection-only: 2/2 boxes matched, mean center drift about 0.24 px, mean corner drift about 0.45 px.
- `ch_doc_server.png` DBPostProcess: 2/2 candidates matched with zero geometry drift after output rounding.
- `test_letterbox_like.jpg`: 2/2 lines matched, character accuracy about 0.994.
- `test_without_det.jpg`: 1/1 line matched, exact text match, mean center drift about 0.09 px.
- `text_vertical_words.png`: 3/3 lines matched, exact text match.
- `latin.jpg`: 1/1 line matched, exact text match.
- `return_word_debug.jpg` with cls enabled: 5/5 lines matched, exact text match, mean center drift about 0.68 px.
- `issue_170.png`: 1/1 line matched, exact text match; the fixture uses a local corner-drift tolerance of 8 px because the current Rust polygon corners differ slightly more than the global 6 px gate while the center and text remain stable.
- `en_rec.jpg` recognition-only: cls enabled and disabled both match the long English line exactly.
- `text_rec.jpg` recognition-only: cls enabled and disabled both recognize `韩国小馆`.
- `text_cls.jpg`: cls enabled recognizes the rotated crop, `--no-cls` leaves it unrecognized.

## Known Differences

### Symbol Normalization

Observed on `ch_en_num.jpg` and `text_det.jpg`.

Examples:

- Python: `-40℃深度防冻不结冰`
- Rust: `-40C深度防冻不结冰`
- Python: `券后价¥`
- Rust: `券后价￥`

Impact:

- Line count and box matching are stable.
- Character accuracy remains high.
- Exact text ratio is below 1.0.

Likely causes:

- Recognition preprocessing and decoding are close but not bit-identical.
- Dictionary/model output differences around visually similar symbols.

Next step:

- Keep character accuracy as the main text parity metric.
- Add focused symbol fixtures if this becomes user-visible.

### Transparent Text Images

Observed on `black_font_color_transparent.png` and `white_font_color_transparent.png`.

Current candidate behavior:

- Python detects short lines such as `中国`, `我`, and `是`.
- Rust now matches `black_font_color_transparent.png` in the full OCR pipeline and DBPostProcess gate after alpha-channel images are composited onto a high-contrast background.
- `white_font_color_transparent.png` now matches the main three text lines in the full OCR pipeline, but Python still emits an additional low-confidence `_` line with score about 0.525 that Rust does not emit.
- `white_font_color_transparent.png` now matches Python's 5 detection candidates after the near-threshold DB score tolerance, but still has mean corner drift about 10 px.
- `white_font_color_transparent.png` also still does not pass DBPostProcess parity because the matched boxes exceed the strict 5 px mean corner-drift gate.

Impact:

- The black-font case is now a strict e2e and DBPostProcess regression.
- The white-font case remains useful for low-confidence and small transparent candidate work, but is not a strict gate yet.

Next step:

- Investigate whether the low-confidence `_` candidate in the white-font image is desirable enough to preserve before adding it as a strict fixture.

### Slanted Text Without Classification

Observed on `return_word_debug.jpg` with `--no-cls`.

Current candidate behavior:

- Python no-cls output keeps the second line as `24`.
- Rust no-cls output recognizes the same crop as `24H专业健身|本座3F1`.
- The cls-enabled fixture is strict and stable: 5/5 lines matched, exact text match.

Impact:

- This image is a good strict gate for the default cls-enabled pipeline.
- It is not a good no-cls parity gate until the intended crop orientation and recognition behavior are understood.

Next step:

- Revisit after crop orientation parity is tightened for slanted detector boxes.

### Tiny Border Text Without Classification

Observed on `ch_doc_server.png`.

Current candidate behavior:

- Rust now matches Python's cls-enabled e2e output after edge-near perspective crops replicate border pixels like OpenCV.
- The image is now also a strict detection-only geometry gate: 2/2 boxes matched with mean center drift about 0.24 px.
- With cls disabled, Python recognizes the tiny top-border crop as `1113C`; Rust recognizes it as `1115C`.

Impact:

- The image is a strict cls-enabled e2e and detection-only geometry gate for tiny edge text.
- It is not a strict no-cls e2e gate yet.

Next step:

- Investigate no-cls crop recognition drift before adding stricter variants.

### Dense Document Text

Observed on `check_return_word_len.jpeg`.

Current candidate behavior:

- The image is now a strict detection-only geometry gate: 28/28 boxes matched, mean center drift about 0.58 px, and mean corner drift about 0.92 px.
- Text parity is not strict enough yet: exact text ratio about 0.57 and character accuracy about 0.934, below the current global 0.96 gate.

Impact:

- Detection layout is close enough to be useful, but recognition drift across many dense small text lines would make it a weak strict e2e gate.
- The detection-only fixture protects the stable layout behavior while recognition remains out of the strict gate.

Next step:

- Use this image while investigating dense small-text recognition differences, then promote it once text parity improves or a focused tolerance is justified.

## Resolved Differences

### EXIF-Oriented Text Image

Observed on `img_exif_orientation.jpg`.

Python normalizes EXIF orientation through `ImageOps.exif_transpose`. Rust now reads decoder orientation metadata and applies it before OCR. The image is a strict e2e fixture with cls enabled and disabled.

### Tiny Edge Text With Classification

Observed on `ch_doc_server.png`.

Python uses OpenCV `BORDER_REPLICATE` for perspective crops. Rust now pads edge-near crops with replicated border pixels before warping, which makes the tiny top-border text recognizable with cls enabled.

### Tiny Edge DBPostProcess Candidate Filtering

Observed on `ch_doc_server.png`.

Rust now rounds output boxes before the final integer-size filter, uses a small score tolerance for near-threshold DB candidates, and drops only near-square 4 px micro boxes produced by the convex offset approximation. This matches Python's 2 DBPostProcess candidates exactly on the fixture while preserving the valid 31x4 top-border text candidate.

### Letterbox-Like Long Lines

Observed on `test_letterbox_like.jpg`.

Previous Rust output split the image into 11 smaller fragments. The root cause was not DBPostProcess itself: the DB fixture already matched Python. The pipeline was missing Python's vertical padding before detection. After adding the same padding step, Rust now outputs 2 lines like Python.

### Vertical Text

Observed on `text_vertical_words.png`.

Previous Rust output detected similar boxes but recognized poor text. The root cause was crop rotation direction: Python `np.rot90` rotates counter-clockwise, while the Rust code used the clockwise image rotation helper. Rust now rotates tall crops counter-clockwise and matches Python on the fixture.

### Latin Paragraph Image

Observed on `latin.jpg`.

Previous Rust output split the paragraph into multiple fragments. The same vertical padding fix used for `test_letterbox_like.jpg` makes Rust output 1 line like Python.

## Policy

- Add images to strict e2e gates only when current behavior is stable and the failure mode is actionable.
- Record known gaps here instead of silently excluding difficult images.
- Prefer metrics that reflect the real risk:
  - character accuracy for symbol-level recognition differences
  - line count and matching for detection/layout differences
  - center/corner drift for geometry differences
