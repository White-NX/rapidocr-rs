# rapidocr-rs Parity Gaps

This file records known differences between the current Rust MVP and Python RapidOCR. These are not ignored forever; they are kept out of strict e2e gates until the expected behavior is understood and the implementation is closer to Python parity.

## Current Strong Gates

The current e2e parity fixtures cover:

- `ch_en_num.jpg`
- `text_det.jpg`
- `ch_en_num.jpg` and `text_det.jpg` as detection-only geometry checks
- `check_return_word_len.jpeg` as a dense-text detection-only geometry check
- `arabic.png`, `cyrillic.png`, `devanagari.jpg`, `japan.jpg`, and `korean.jpg` as cross-language detection-only geometry checks
- `ta.png`, `th_rec.jpg`, `te.png`, and `eslav.jpg` as additional script/layout detection-only geometry checks
- `te.png` with cls enabled and disabled as a default-model full-pipeline parity check
- `en.jpg`
- `empty_black.jpg`
- `short.png`
- `black_font_color_transparent.png`
- `white_font_color_transparent.png` as a detection-only geometry check with local corner-drift tolerance
- `img_exif_orientation.jpg`
- `ch_doc_server.png` with cls enabled and detection-only
- `test_letterbox_like.jpg`
- `test_without_det.jpg`
- `text_vertical_words.png`
- `latin.jpg`
- `return_word_debug.jpg` with cls enabled
- `en_rec.jpg`, `el_rec.jpg`, and `devanagari_rec.png` as recognition-crop detection-only geometry checks
- `en_rec.jpg` as a recognition-only cls/no-cls long English line check
- `el_rec.jpg` as a recognition-only cls/no-cls Greek-script default-model check
- `devanagari_rec.png` as a recognition-only no-cls default-model check
- `text_rec.jpg` as a recognition-only cls/no-cls normal-crop check
- `text_cls.jpg` as a recognition-only cls/no-cls 180-degree crop check
- `text_cls.jpg` as a Rust cls/no-cls golden

The current DBPostProcess parity fixtures additionally cover `black_font_color_transparent.png`, `white_font_color_transparent.png`, `return_word_debug.jpg`, `short.png`, `test_without_det.jpg`, `ch_doc_server.png`, `check_return_word_len.jpeg`, `latin.jpg`, `img_exif_orientation.jpg`, `en_rec.jpg`, `el_rec.jpg`, `devanagari_rec.png`, `arabic.png`, `cyrillic.png`, `devanagari.jpg`, `japan.jpg`, `korean.jpg`, `ta.png`, `th_rec.jpg`, `te.png`, and `eslav.jpg`.

Current representative metrics:

- `ch_en_num.jpg` and `text_det.jpg`: 18/18 lines matched, character accuracy about 0.976, mean center drift about 1.23 px.
- `ch_en_num.jpg` and `text_det.jpg` detection-only: 18/18 boxes matched, mean center drift about 1.23 px, mean corner drift about 1.45 px.
- `check_return_word_len.jpeg` detection-only: 28/28 boxes matched, mean center drift about 0.58 px, mean corner drift about 0.92 px.
- Cross-language detection-only: `arabic.png` 2/2, `cyrillic.png` 4/4, `devanagari.jpg` 4/4, `japan.jpg` 7/7, and `korean.jpg` 6/6 boxes matched with mean center drift below 0.70 px.
- `en.jpg`: 5/5 lines matched, exact text match, mean center drift about 0.21 px.
- `empty_black.jpg`: 0/0 lines matched.
- `short.png`: 0/0 lines matched.
- `black_font_color_transparent.png`: 3/3 lines matched, exact text match, mean center drift about 1.03 px.
- `white_font_color_transparent.png` detection-only: 5/5 boxes matched, mean center drift about 1.36 px; it uses a local corner-drift tolerance for documented offset-geometry differences.
- `white_font_color_transparent.png` DBPostProcess: 5/5 candidates matched, mean center drift about 1.37 px; it uses local corner/size drift tolerances for documented offset-geometry differences.
- `img_exif_orientation.jpg`: 1/1 line matched, exact text match, mean center drift about 0.45 px.
- `ch_doc_server.png` with cls enabled: 2/2 lines matched, exact text match, mean corner drift about 0.45 px.
- `ch_doc_server.png` detection-only: 2/2 boxes matched, mean center drift about 0.24 px, mean corner drift about 0.45 px.
- `ch_doc_server.png` DBPostProcess: 2/2 candidates matched with zero geometry drift after output rounding.
- `check_return_word_len.jpeg` DBPostProcess: 28/28 candidates matched, mean center drift about 0.63 px, mean corner drift about 0.94 px.
- Cross-language DBPostProcess: `arabic.png` 2/2, `cyrillic.png` 4/4, `devanagari.jpg` 4/4, `japan.jpg` 7/7, and `korean.jpg` 6/6 candidates matched with mean center drift below 0.66 px.
- Additional script/layout detection-only: `ta.png` 2/2, `th_rec.jpg` 1/1, `te.png` 1/1, and `eslav.jpg` 1/1 boxes matched with mean center drift at or below 0.50 px.
- Additional script/layout DBPostProcess: `ta.png` 2/2, `th_rec.jpg` 1/1, `te.png` 1/1, and `eslav.jpg` 1/1 candidates matched with mean corner drift at or below 1.21 px.
- Recognition-crop detection-only: `en_rec.jpg` 1/1, `el_rec.jpg` 3/3, and `devanagari_rec.png` 2/2 boxes matched with mean corner drift at or below 0.75 px.
- Recognition-crop DBPostProcess: `en_rec.jpg` 1/1, `el_rec.jpg` 3/3, and `devanagari_rec.png` 2/2 candidates matched with mean corner drift at or below 0.50 px.
- `te.png` full e2e: cls enabled and disabled both match Python's default-model output `.` exactly.
- `test_letterbox_like.jpg`: 2/2 lines matched, character accuracy about 0.994.
- `test_without_det.jpg`: 1/1 line matched, exact text match, mean center drift about 0.09 px.
- `text_vertical_words.png`: 3/3 lines matched, exact text match.
- `latin.jpg`: 1/1 line matched, exact text match.
- `latin.jpg` DBPostProcess: 1/1 candidate matched, mean corner drift about 1.21 px.
- `img_exif_orientation.jpg` DBPostProcess: 1/1 candidate matched, mean corner drift about 0.96 px.
- `return_word_debug.jpg` with cls enabled: 5/5 lines matched, exact text match, mean center drift about 0.68 px.
- `issue_170.png`: 1/1 line matched, exact text match; the fixture uses a local corner-drift tolerance of 8 px because the current Rust polygon corners differ slightly more than the global 6 px gate while the center and text remain stable.
- `en_rec.jpg` recognition-only: cls enabled and disabled both match the long English line exactly.
- `el_rec.jpg` recognition-only: cls enabled and disabled both match `Ωραíο αρ σμεα.` exactly.
- `devanagari_rec.png` recognition-only no-cls: both Python and Rust output `H`; this is a default-model parity gate, not a language correctness claim.
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
- `white_font_color_transparent.png` now matches Python's 5 detection-only boxes and 5 DBPostProcess candidates after the near-threshold DB score tolerance. It is a strict detection/DB gate with local geometry tolerances: mean corner drift is about 10 px and DB mean size drift about 5.2 px.
- `white_font_color_transparent.png` still does not pass full e2e parity because Python emits the additional low-confidence `_` line while Rust does not.

Impact:

- The black-font case is a strict e2e and DBPostProcess regression.
- The white-font case now protects detection candidate count and DB score behavior, but the full OCR output remains outside strict e2e gates.

Next step:

- Investigate whether the low-confidence `_` candidate in the white-font image is desirable enough to preserve before adding it as a strict full e2e fixture.

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
- It is also now a strict DBPostProcess gate: 28/28 candidates matched, mean center drift about 0.63 px, and mean corner drift about 0.94 px.
- Text parity is not strict enough yet: exact text ratio about 0.57 and character accuracy about 0.934, below the current global 0.96 gate.

Impact:

- Detection layout is close enough to be useful, but recognition drift across many dense small text lines would make it a weak strict e2e gate.
- The detection-only and DBPostProcess fixtures protect the stable layout behavior while recognition remains out of the strict gate.

Next step:

- Use this image while investigating dense small-text recognition differences, then promote it once text parity improves or a focused tolerance is justified.

### Language-Specific Recognition Crops

Observed on `devanagari_rec.png`, `th_rec.jpg`, `ta.png`, `te.png`, and `eslav.jpg`.

Current candidate behavior:

- `devanagari_rec.png` no-cls is now a strict recognition-only gate because Rust and Python both output `H`.
- `devanagari_rec.png` with cls enabled is not strict: Python emits `和5`, while Rust currently emits no line after the cls path.
- `th_rec.jpg` detection-only and DBPostProcess geometry are now strict 1/1 gates, but recognition-only is not strict: Python emits `nsuwnuuzinunavnlunnaunuiula`, while Rust emits `nunuuziunavnunnaunuiul`; character accuracy is about 0.815.
- `te.png` is now a strict full-pipeline gate because Rust and Python both output `.` with score drift under the current gate.
- `ta.png` full-pipeline candidate is not strict: detection geometry is stable, but text parity is poor with character accuracy about 0.40.
- `eslav.jpg` full-pipeline candidate is not strict yet: text and geometry match, but recognition score drift is about 0.112, above the current 0.08 gate.

Impact:

- The default PP-OCRv6 small recognition model is useful for parity checks on these crops, but not for claiming language correctness.
- Cls-sensitive behavior on non-default-language crops still needs tighter investigation before promotion.

Next step:

- Revisit after model-matrix work adds language-specific recognition dictionaries and models, or after cls crop handling differences are narrowed further.

### Recognition Crop Detection Candidates

Observed on `text_rec.jpg` and `text_cls.jpg` when evaluated as detection-only and DBPostProcess candidates.

Current candidate behavior:

- Python emits no detection candidate for these cropped recognition images.
- Rust currently emits one detection candidate for each image.
- The existing recognition-only gates remain strict and useful; the detector/DB candidate count mismatch is specific to running the detector on these already-cropped inputs.

Impact:

- `text_rec.jpg` and `text_cls.jpg` should stay out of strict detection-only and DBPostProcess gates until the intended behavior is clearer.
- The rec-only fixtures continue to cover normal and rotated crop recognition behavior.

Next step:

- Revisit if recognition-crop detector behavior becomes part of the supported surface, or while tightening near-threshold DB candidate filtering.

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
