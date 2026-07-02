# rapidocr-rs

Rust migration workspace for RapidOCR.

## Phase 1 Goal

Build an ONNX Runtime based Rust OCR core that can run the default detection, optional direction classification, and recognition flow.

This phase is successful when:

- The Rust core can run `det -> rec` with ONNX models.
- Direction classification can be enabled in the default pipeline or removed with a CLI/config switch.
- Outputs are comparable with the Python implementation on the existing external Python RapidOCR test images.
- Detection boxes, recognized text, and confidence scores are covered by golden-output tests.
- The implementation has a small CLI for local validation.
- The design leaves room for a later Python binding, but does not require it in phase 1.

## Initial Scope

In scope:

- Image loading and preprocessing.
- DB text detection preprocessing and postprocessing.
- Text crop and perspective transform.
- Optional text direction classification and 180-degree crop rotation.
- CTC text recognition decoding.
- ONNX Runtime inference backend.
- Golden parity tests against the Python implementation.

Out of scope for phase 1:

- Paddle, PyTorch, OpenVINO, TensorRT, and MNN backends.
- Full Python package replacement.
- Visualization parity.
- Training or model conversion.
- Mobile packaging.

## Workspace Layout

```text
rapidocr-rs/
  crates/
    rapidocr-core/  # OCR pipeline and backend abstraction
    rapidocr-cli/   # Thin command-line wrapper for validation
```

## Current Status

The workspace currently has a runnable `det -> optional cls -> rec` prototype:

```powershell
$env:RAPIDOCR_PYTHON_REPO = "D:\projects\RapidOCR"
cargo run -p rapidocr-cli -- --image "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\ch_en_num.jpg" --model-dir models
```

Skip cls entirely:

```powershell
cargo run -p rapidocr-cli -- --image "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\ch_en_num.jpg" --model-dir models --no-cls
```

Run recognition on the full image without detection:

```powershell
cargo run -p rapidocr-cli -- --image "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\ch_en_num.jpg" --model-dir models --no-det
```

Run detection only:

```powershell
cargo run -p rapidocr-cli -- --image "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\ch_en_num.jpg" --model-dir models --no-rec
```

Write a default TOML config:

```powershell
cargo run -p rapidocr-cli -- --write-default-config config\ppocrv6-small.toml --model-dir models
```

Run with a TOML config:

```powershell
cargo run -p rapidocr-cli -- --image "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\ch_en_num.jpg" --config config\ppocrv6-small.toml
```

## Public API And Configuration

The library API and CLI use the same `RapidOcrConfig` model. The default model set is exposed as data through `PPOCRV6_SMALL`, and model cache behavior is explicit through `ModelCache` and `ModelDownloadMode`.
The CLI applies pipeline overrides before preparing the model cache, so disabled stages do not require their model files.

```rust
use rapidocr_core::{
    config::PipelineConfig,
    model::{ModelCache, ModelDownloadMode, PPOCRV6_SMALL},
    RapidOcr,
};

let cache = ModelCache::new("models");
cache.ensure_model_set(&PPOCRV6_SMALL, ModelDownloadMode::Missing)?;

let cfg = cache
    .config_for(&PPOCRV6_SMALL)
    .with_pipeline(PipelineConfig::without_cls());
let mut ocr = RapidOcr::from_config(cfg)?;
let output = ocr.run_path("D:/projects/RapidOCR/python/tests/test_files/ch_en_num.jpg")?;
```

The generated TOML contains a `[pipeline]` section:

```toml
[pipeline]
use_det = true
use_cls = true
use_rec = true
```

`use_det = false` treats the input image as one recognition crop. `use_rec = false` runs detection only and returns boxes with empty text and score `0.0`. `use_cls = true` requires recognition because cls is only used to rotate recognition crops.

Missing images, model files, dictionaries, invalid config values, and ONNX loading failures include the relevant path and stage in the error message.
Image loading applies file EXIF orientation and composites alpha-channel images onto a high-contrast background before OCR, matching the Python input path for the covered parity fixtures.
Perspective crops near image edges use replicated border pixels before warping, matching Python's OpenCV crop behavior for tiny edge text.

The detector postprocess is still an approximation of Python's `DBPostProcess`. It uses thresholding, dilation, connected components, boundary extraction, convex hull, a rotating-calipers style minimum-area rectangle, Python-style point ordering, polygon mean score filtering, pure Rust polygon area/perimeter metrics, convex polygon offset for unclip expansion, stricter size filtering, and perspective crop. This is enough to validate ONNX Runtime inference, recognition decoding, and the end-to-end CLI, but it is not yet full OpenCV/pyclipper parity.

## DB Postprocess Parity

The DB postprocess is split into `rapidocr-core/src/db_postprocess.rs` so it can be tested without running ONNX inference.

Generate the Python parity fixtures from the standalone Rust repository root. The Python RapidOCR repo is external and must be explicit:

```powershell
$env:RAPIDOCR_PYTHON_REPO = "D:\projects\RapidOCR"
python .\tools\export_db_fixture.py
```

Run the Rust fixture test:

```powershell
cargo test -p rapidocr-core db_postprocess
```

To evaluate DBPostProcess candidates without committing them, export to a temporary directory and point the test at it:

```powershell
python .\tools\export_db_fixture.py --out-dir target\db_candidates --image python\tests\test_files\short.png
$env:RAPIDOCR_DB_FIXTURE_ROOT = "D:\projects\rapidocr-rs\target\db_candidates"
cargo test -p rapidocr-core db_postprocess -- --nocapture
```

The current fixtures cover:

- `ch_en_num.jpg`
- `text_det.jpg`
- `en.jpg`
- `test_letterbox_like.jpg`
- `test_without_det.jpg`
- `text_vertical_words.png`
- `latin.jpg`
- `img_exif_orientation.jpg`
- `empty_black.jpg`
- `issue_170.png`
- `short.png`
- `return_word_debug.jpg`
- `black_font_color_transparent.png`
- `white_font_color_transparent.png`
- `ch_doc_server.png`
- `check_return_word_len.jpeg`
- `arabic.png`
- `cyrillic.png`
- `devanagari.jpg`
- `japan.jpg`
- `korean.jpg`
- `ta.png`
- `th_rec.jpg`
- `te.png`
- `eslav.jpg`

The test checks candidate count, center-distance drift, score drift, corner drift, and width/height drift against Python's `DBPostProcess` output. Use `-- --nocapture` to print the current metrics. Fixtures may include local `tolerances` for documented geometry drift.

## End-to-End Golden Tests

End-to-end fixtures live under `fixtures/e2e`.

Generate Python parity fixtures from the standalone Rust repository root:

```powershell
$env:RAPIDOCR_PYTHON_REPO = "D:\projects\RapidOCR"
python .\tools\export_e2e_fixture.py
```

Run the e2e metric test:

```powershell
cargo test -p rapidocr-core e2e_output_tracks_golden_metrics -- --nocapture
```

To evaluate e2e candidates without committing them, export to a temporary directory and point the test at it:

```powershell
python .\tools\export_e2e_fixture.py --out-dir target\e2e_candidates --image python\tests\test_files\en_rec.jpg --pipeline rec-only
$env:RAPIDOCR_E2E_FIXTURE_ROOT = "D:\projects\rapidocr-rs\target\e2e_candidates"
cargo test -p rapidocr-core e2e_output_tracks_golden_metrics -- --nocapture
```

Use `--pipeline det-only` to evaluate detection-only geometry candidates.

The current e2e fixtures cover:

- `ch_en_num.jpg` with cls enabled, cls disabled, and detection-only.
- `text_det.jpg` with cls enabled, cls disabled, and detection-only.
- `check_return_word_len.jpeg` as a dense-text detection-only geometry check.
- `arabic.png`, `cyrillic.png`, `devanagari.jpg`, `japan.jpg`, and `korean.jpg` as cross-language detection-only geometry checks.
- `ta.png`, `th_rec.jpg`, `te.png`, and `eslav.jpg` as additional script/layout detection-only geometry checks.
- `te.png` with cls enabled and disabled as a default-model full-pipeline parity check.
- `en.jpg` with cls enabled and disabled.
- `empty_black.jpg` with cls enabled and disabled.
- `short.png` with cls enabled and disabled.
- `black_font_color_transparent.png` with cls enabled and disabled.
- `white_font_color_transparent.png` as a detection-only geometry check with documented local corner-drift tolerance.
- `img_exif_orientation.jpg` with cls enabled and disabled.
- `ch_doc_server.png` with cls enabled and detection-only.
- `test_letterbox_like.jpg` with cls enabled and disabled.
- `test_without_det.jpg` with cls enabled and disabled.
- `return_word_debug.jpg` with cls enabled.
- `text_vertical_words.png` with cls enabled and disabled.
- `latin.jpg` with cls enabled and disabled.
- `issue_170.png` with cls enabled and disabled.
- `en_rec.jpg` recognition-only with cls enabled and disabled.
- `el_rec.jpg` recognition-only with cls enabled and disabled.
- `devanagari_rec.png` recognition-only with cls disabled.
- `text_rec.jpg` recognition-only with cls enabled and disabled.
- `text_cls.jpg` recognition-only with cls enabled and disabled.
- `text_cls.jpg` as a Rust golden for the cls/no-cls pipeline switch.

The test checks line count, nearest-center matching, exact text ratio, character accuracy, score drift, center drift, and corner drift. Detection-only fixtures skip recognition text/score gates and keep the count and geometry gates. It requires downloaded models in `models` and Python test images from `RAPIDOCR_PYTHON_REPO`.
Fixtures use the default metric gates unless the JSON contains a `tolerances` object for a known, documented geometry difference.
Fixtures may also contain a `pipeline` object with `use_det`, `use_cls`, and `use_rec` for non-default pipeline coverage such as recognition-only cls behavior and detection-only geometry checks.

Instead of setting the environment variable every time, create a local ignored `config/local.toml`:

```toml
[parity]
python_repo = "D:/projects/RapidOCR"
```

Known gaps that are not yet strict gates are tracked in `parity-gaps.md`.

## Benchmark Commands

Run the Rust CLI in a hot loop, reusing the loaded OCR pipeline:

```powershell
cargo run --release -p rapidocr-cli -- --image "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\ch_en_num.jpg" --model-dir models --no-download --repeat 20 --quiet
```

Compare Rust CLI and Python RapidOCR hot-loop timings:

```powershell
python .\tools\bench_e2e.py --repeat 20 --image python\tests\test_files\ch_en_num.jpg
```

Use `--no-cls` to benchmark the pipeline without cls.
Use `--out target\benchmark.md` to write the measured rows as Markdown. The curated current baseline is recorded in `benchmark-baseline.md`.
