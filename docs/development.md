# Development

This page collects local development, parity, and benchmark workflows for `rapidocr-rs`.

## Workspace Layout

```text
rapidocr-rs/
  crates/
    rapidocr-core/  # OCR pipeline, model registry, configuration, and backend abstraction
    rapidocr-cli/   # Workspace CLI wrapper for validation and benchmarking
  fixtures/
    db_postprocess/ # DBPostProcess parity fixtures exported from Python RapidOCR
    e2e/            # End-to-end golden fixtures exported from Python RapidOCR
  tools/            # Fixture export and benchmark helper scripts
```

## Fast Local Gate

The default Rust checks do not require downloaded ONNX models or the external Python RapidOCR repository:

```powershell
cargo fmt --check
cargo check
cargo test
```

Run this path before commits that touch Rust code, configuration, examples, or docs with code snippets.

## CLI Development

Run the default full pipeline:

```powershell
cargo run -p rapidocr-cli -- --image path\to\image.png --model-dir models
```

Skip direction classification:

```powershell
cargo run -p rapidocr-cli -- --image path\to\image.png --model-dir models --no-cls
```

Run recognition on the whole input image:

```powershell
cargo run -p rapidocr-cli -- --image path\to\crop.png --model-dir models --no-det
```

Run detection only:

```powershell
cargo run -p rapidocr-cli -- --image path\to\image.png --model-dir models --no-rec
```

Write a default TOML config:

```powershell
cargo run -p rapidocr-cli -- --write-default-config config\ppocrv6-small.toml --model-dir models
```

Run from a TOML config:

```powershell
cargo run -p rapidocr-cli -- --image path\to\image.png --config config\ppocrv6-small.toml
```

The generated TOML contains a `[pipeline]` section:

```toml
[pipeline]
use_det = true
use_cls = true
use_rec = true
```

`use_det = false` treats the input image as one recognition crop. `use_rec = false` runs detection only and returns boxes with empty text and score `0.0`. `use_cls = true` requires recognition because cls is only used to rotate recognition crops.

## Library API

`rapidocr-core` exposes the same configuration model used by the CLI:

- `RapidOcr` runs the OCR pipeline.
- `RapidOcrConfig` is TOML-compatible.
- `available_model_sets` and `model_set_by_name` expose the model registry.
- `ModelCache` and `ModelDownloadMode` make model download behavior explicit.

Pipeline overrides are applied before model cache preparation in the CLI, so disabled stages do not require their model files.

## Model Registry Smoke Checks

The current registry contains:

- `ppocrv6-small` (default)
- `ppocrv6-tiny`
- `ppocrv6-medium`
- `ppocrv4-en-mobile`
- `ppocrv5-en-mobile`
- `ppocrv5-ch-server`

Run non-default recognition-only smoke checks when changing model registration or cache logic:

```powershell
$env:RAPIDOCR_PYTHON_REPO = "C:\path\to\RapidOCR"
cargo run -p rapidocr-cli -- --model-set ppocrv4-en-mobile --image "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\en_rec.jpg" --model-dir models --no-det --no-cls
cargo run -p rapidocr-cli -- --model-set ppocrv5-en-mobile --image "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\en_rec.jpg" --model-dir models --no-det --no-cls
cargo test -p rapidocr-core non_default_model_sets_run_rec_only_smoke -- --ignored --nocapture
```

## Python RapidOCR Checkout

Parity fixture scripts expect an external Python RapidOCR checkout. Set it explicitly:

```powershell
$env:RAPIDOCR_PYTHON_REPO = "C:\path\to\RapidOCR"
```

Instead of setting the environment variable every time, create a local ignored `config/local.toml`:

```toml
[parity]
python_repo = "C:/path/to/RapidOCR"
```

## DBPostProcess Parity

The detector postprocess lives in `rapidocr-core/src/db_postprocess.rs` and can be tested without running ONNX inference.

Export fixtures from the standalone Rust repository root:

```powershell
$env:RAPIDOCR_PYTHON_REPO = "C:\path\to\RapidOCR"
python .\tools\export_db_fixture.py
```

Run the Rust fixture test:

```powershell
cargo test -p rapidocr-core db_postprocess
```

To evaluate DBPostProcess candidates without committing generated fixtures, export to a temporary directory and point the test at it:

```powershell
python .\tools\export_db_fixture.py --out-dir target\db_candidates --image python\tests\test_files\short.png
$env:RAPIDOCR_DB_FIXTURE_ROOT = "C:\path\to\rapidocr-rs\target\db_candidates"
cargo test -p rapidocr-core db_postprocess -- --nocapture
```

The test checks candidate count, center-distance drift, score drift, corner drift, and width/height drift against Python RapidOCR's `DBPostProcess` output. Fixtures may include local `tolerances` for documented geometry drift.

## End-to-End Golden Tests

End-to-end fixtures live under `fixtures/e2e`.

Generate Python parity fixtures:

```powershell
$env:RAPIDOCR_PYTHON_REPO = "C:\path\to\RapidOCR"
python .\tools\export_e2e_fixture.py
```

Run the ignored e2e metric test:

```powershell
cargo test -p rapidocr-core e2e_output_tracks_golden_metrics -- --ignored --nocapture
```

To evaluate e2e candidates without committing generated fixtures, export to a temporary directory and point the test at it:

```powershell
python .\tools\export_e2e_fixture.py --out-dir target\e2e_candidates --image python\tests\test_files\en_rec.jpg --pipeline rec-only
$env:RAPIDOCR_E2E_FIXTURE_ROOT = "C:\path\to\rapidocr-rs\target\e2e_candidates"
cargo test -p rapidocr-core e2e_output_tracks_golden_metrics -- --ignored --nocapture
```

Use `--pipeline det-only` to evaluate detection-only geometry candidates.

The e2e test checks line count, nearest-center matching, exact text ratio, character accuracy, score drift, center drift, and corner drift. Detection-only fixtures skip recognition text and score gates. Fixtures use the default metric gates unless the JSON contains a `tolerances` object for a known, documented difference. Fixtures may also contain a `pipeline` object with `use_det`, `use_cls`, and `use_rec`.

Known parity gaps that are not strict gates are tracked in [../parity-gaps.md](../parity-gaps.md).

## Benchmark Commands

Run the Rust CLI in a hot loop, reusing the loaded OCR pipeline:

```powershell
cargo run --release -p rapidocr-cli -- --image "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\ch_en_num.jpg" --model-dir models --no-download --repeat 20 --quiet
```

Emit the Rust timing summary as JSON:

```powershell
cargo run --release -p rapidocr-cli -- --image "$env:RAPIDOCR_PYTHON_REPO\python\tests\test_files\ch_en_num.jpg" --model-dir models --no-download --repeat 20 --benchmark-json
```

Compare Rust CLI and Python RapidOCR hot-loop timings:

```powershell
python .\tools\bench_e2e.py --repeat 20 --image python\tests\test_files\ch_en_num.jpg
```

Use `--no-cls` to benchmark the pipeline without cls. Use `--out target\benchmark.md` to write measured rows as Markdown.

The benchmark script records model load time, end-to-end hot-loop time, peak RSS when `psutil` is installed, and stage summaries for det/cls/rec. Rust rows also include `det_postprocess_ms` from `OcrTimings`; the Python timing surface does not split DB postprocess separately.

The curated current baseline is recorded in [../benchmark-baseline.md](../benchmark-baseline.md).

## Error And Image Handling Notes

Missing images, model files, dictionaries, invalid config values, and ONNX loading failures include the relevant path and stage in the error message.

Image loading applies file EXIF orientation and composites alpha-channel images onto a high-contrast background before OCR. Perspective crops near image edges use replicated border pixels before warping, matching Python's OpenCV crop behavior for tiny edge text in the covered parity fixtures.
