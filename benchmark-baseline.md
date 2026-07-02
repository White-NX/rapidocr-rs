# rapidocr-rs Benchmark Baseline

This file records repeatable benchmark results for the current default ONNX pipeline. The numbers are local measurements, not release guarantees.

## Source Snapshot

- Recorded at: 2026-07-02 23:30 Asia/Shanghai.
- Git HEAD: `a8d48ab50d834d68d3801ca7a5bd705a85db8f3b`.
- Git state: dirty working tree included. The run includes the current Phase 4 fixture updates and Phase 5 timing/copy changes.
- Rust: `rustc 1.90.0 (1159e78c4 2025-09-14)`, `cargo 1.90.0 (840b83a10 2025-07-30)`.
- Python: `Python 3.12.10`.
- Platform: `Windows-11-10.0.26220-SP0`.
- Rust repo: `D:\projects\rapidocr-rs`.
- Python parity repo: `D:\projects\RapidOCR`.
- Model dir: `D:\projects\rapidocr-rs\models`.

## Command

```powershell
$env:RAPIDOCR_PYTHON_REPO = "D:\projects\RapidOCR"
python .\tools\bench_e2e.py --repeat 5 --image python\tests\test_files\ch_en_num.jpg --out target\benchmark_phase5_stage_cls_current.md
python .\tools\bench_e2e.py --repeat 5 --image python\tests\test_files\ch_en_num.jpg --no-cls --out target\benchmark_phase5_stage_no_cls_current.md
```

The Rust path builds and runs the release `rapidocr-cli` with `--benchmark-json`, measures `RapidOcr::new` as `model_load_ms`, then reuses the loaded OCR pipeline for repeated `run_path_timed` calls. The Python path measures a `RapidOCR` instance after one warm-up call. Both use the same model directory.

Rust stage values come from `OcrTimings`. `det_ms` is the sum of pipeline preprocessing, detection preprocessing, detection ONNX inference, and DB postprocess. `det_postprocess_ms` is split out for Rust only. Python stage values come from RapidOCR's `elapse_list`, which reports det/cls/rec but does not split DB postprocess. `peak_rss_mb` is reported when `psutil` is installed.

## Stage-Level Hot Loop

| image | use_cls | runner | model_load_ms | mean_ms | min_ms | max_ms | peak_rss_mb | det_ms | cls_ms | rec_ms | det_postprocess_ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `python/tests/test_files/ch_en_num.jpg` | true | `rust_cli_hot` | 215.055 | 315.491 | 287.821 | 332.010 | 400.6 | 143.563 | 15.696 | 149.048 | 10.808 |
| `python/tests/test_files/ch_en_num.jpg` | true | `python_hot` | 415.457 | 322.002 | 315.972 | 327.253 | 177.0 | 152.016 | 14.216 | 149.638 |  |
| `python/tests/test_files/ch_en_num.jpg` | false | `rust_cli_hot` | 182.304 | 303.417 | 297.681 | 318.419 | 390.7 | 143.019 | 0.000 | 152.688 | 11.036 |
| `python/tests/test_files/ch_en_num.jpg` | false | `python_hot` | 418.765 | 304.932 | 300.497 | 318.985 | 175.5 | 152.962 |  | 145.816 |  |

## Bottleneck Notes

- End-to-end hot-loop latency is similar on this image. Rust measured `315.491 ms` with cls and `303.417 ms` without cls; Python measured `322.002 ms` with cls and `304.932 ms` without cls.
- Recognition and detection dominate the hot loop. On the Rust cls run, `rec_ms` was `149.048 ms` and `det_ms` was `143.563 ms`; `cls_ms` was `15.696 ms`.
- Rust model load time measured lower than Python in these runs, but Rust peak RSS measured higher: about `391-401 MB` for Rust versus `176-177 MB` for Python.
- Rust DB postprocess cost is now measured separately at about `10.8-11.0 ms` on this image. The Python comparison path does not expose an equivalent split.
- The current code removes two avoidable copies from the measured default path: cls rotation now keeps owned crops and only allocates replacement images when a crop is rotated, and recognition decode reads logits by view instead of copying each slice.

## Current Limitations

- This baseline covers one local Windows CPU run on one image. It should not be used as a cross-platform claim.
- RSS is sampled from process memory with `psutil`; short-lived peaks between samples may be missed.
- Python DB postprocess time is not split from detection by the current RapidOCR timing surface.
- Batch-size tuning is not covered. The current numbers exercise the configured cls/rec batch paths for a single image.
- The benchmark script can write Markdown records with `--out`; curated baseline updates should stay in this file.
