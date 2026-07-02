# rapidocr-rs Benchmark Baseline

This file records repeatable benchmark results for the current default ONNX pipeline. The numbers are local measurements, not release guarantees.

## Source Snapshot

- Recorded at: 2026-07-02 13:44 Asia/Shanghai.
- Git state: no initial commit is available; the benchmark is based on the current uncommitted working tree.
- Rust: `rustc 1.90.0 (1159e78c4 2025-09-14)`, `cargo 1.90.0 (840b83a10 2025-07-30)`.
- Python: `Python 3.12.10`.
- Platform: `Windows-11-10.0.26220-SP0`.
- Rust repo: `D:\projects\rapidocr-rs`.
- Python parity repo: `D:\projects\RapidOCR`.
- Model dir: `D:\projects\rapidocr-rs\models`.

## Command

```powershell
$env:RAPIDOCR_PYTHON_REPO = "D:\projects\RapidOCR"
python .\tools\bench_e2e.py --repeat 20 --image python\tests\test_files\ch_en_num.jpg
python .\tools\bench_e2e.py --repeat 20 --image python\tests\test_files\ch_en_num.jpg --no-cls
```

The Rust path measures `rapidocr-cli` with the OCR pipeline loaded once and reused for the repeated `run_path` calls. The Python path measures a `RapidOCR` instance after one warm-up call. Both use the same model directory.

## End-To-End Hot Loop

| image | use_cls | runner | mean_ms | min_ms | max_ms |
| --- | --- | --- | ---: | ---: | ---: |
| `python/tests/test_files/ch_en_num.jpg` | true | `rust_cli_hot` | 314.417 | 276.870 | 343.244 |
| `python/tests/test_files/ch_en_num.jpg` | true | `python_hot` | 310.542 | 303.752 | 318.983 |
| `python/tests/test_files/ch_en_num.jpg` | false | `rust_cli_hot` | 313.492 | 283.389 | 342.926 |
| `python/tests/test_files/ch_en_num.jpg` | false | `python_hot` | 292.073 | 284.538 | 308.403 |

## Current Limitations

- This baseline only covers end-to-end hot-loop latency on one image.
- It does not split detection, classification, recognition, postprocess, model load time, or memory usage.
- The benchmark script can write Markdown records with `--out`, but curated baseline updates should stay in this file.
