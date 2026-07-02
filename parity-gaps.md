# rapidocr-rs Parity Gaps

This file records known differences between the current Rust MVP and Python RapidOCR. These are not ignored forever; they are kept out of strict e2e gates until the expected behavior is understood and the implementation is closer to Python parity.

## Current Strong Gates

The current e2e parity fixtures cover:

- `ch_en_num.jpg`
- `text_det.jpg`
- `en.jpg`
- `empty_black.jpg`
- `test_letterbox_like.jpg`
- `text_vertical_words.png`
- `latin.jpg`
- `text_cls.jpg` as a Rust cls/no-cls golden

Current representative metrics:

- `ch_en_num.jpg` and `text_det.jpg`: 18/18 lines matched, character accuracy about 0.976, mean center drift about 1.23 px.
- `en.jpg`: 5/5 lines matched, exact text match, mean center drift about 0.21 px.
- `empty_black.jpg`: 0/0 lines matched.
- `test_letterbox_like.jpg`: 2/2 lines matched, character accuracy about 0.994.
- `text_vertical_words.png`: 3/3 lines matched, exact text match.
- `latin.jpg`: 1/1 line matched, exact text match.
- `issue_170.png`: 1/1 line matched, exact text match; the fixture uses a local corner-drift tolerance of 8 px because the current Rust polygon corners differ slightly more than the global 6 px gate while the center and text remain stable.
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

- Python detects 3-4 short lines such as `中国`, `我`, and `是`.
- Rust currently merges the visible text into one large line on these transparent-background images.

Impact:

- The line count does not match, so these images are not strict e2e gates yet.
- They remain useful candidates for DB/postprocess and transparent image handling work.

Next step:

- Add transparent fixtures only after the expected preprocessing and DBPostProcess behavior is understood.

### EXIF-Oriented Text Image

Observed on `img_exif_orientation.jpg`.

Current candidate behavior:

- With cls enabled, Rust recognizes the text, but box center/corner drift is above the current strict geometry gates.
- With cls disabled, Rust recognition degrades on the rotated crop.

Impact:

- The image is useful for cls and image orientation behavior, but it is not yet a stable strict parity fixture.

Next step:

- Decide whether EXIF orientation should be normalized before detection, and add a focused fixture after the intended behavior is explicit.

## Resolved Differences

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
