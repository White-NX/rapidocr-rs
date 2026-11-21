# rapidocr-rs Benchmark Baseline

This file records repeatable benchmark results for the current default ONNX pipeline. The numbers are local measurements, not release guarantees.

**The July Rust/Python RSS columns below use different sampling methods and are not comparable peak measurements.** The November RSS audit corrects the measurement and uses the current pipeline. The July timing rows are retained as historical records.

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

In this historical run, the Rust path built and ran the release `rapidocr-cli` with `--benchmark-json`, measured `RapidOcr::new` as `model_load_ms`, then reused the loaded OCR pipeline for repeated `run_path_timed` calls without a warm-up. The Python path measured a `RapidOCR` instance after one warm-up call. Both used the same model directory; the current default models differ from this snapshot.

Rust stage values come from `OcrTimings`. `det_ms` is the sum of pipeline preprocessing, detection preprocessing, detection ONNX inference, and DB postprocess. `det_postprocess_ms` is split out for Rust only. Python stage values come from RapidOCR's `elapse_list`, which reports det/cls/rec but does not split DB postprocess. The historical `peak_rss_mb` values actually use MiB: Rust was sampled every 10 ms, but Python RSS was read only before/after model loading and after each completed inference.

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
- Rust model load time measured lower than Python in these runs. The reported `391-401 MiB` for Rust versus `176-177 MiB` for Python does not establish a peak-RSS difference: the Python readings missed temporary inference allocations, and the runtime configurations differed.
- Rust DB postprocess cost is now measured separately at about `10.8-11.0 ms` on this image. The Python comparison path does not expose an equivalent split.
- The current code removes two avoidable copies from the measured default path: cls rotation now keeps owned crops and only allocates replacement images when a crop is rotated, and recognition decode reads logits by view instead of copying each slice.

## Current Limitations

- This baseline covers one local Windows CPU run on one image. It should not be used as a cross-platform claim.
- The historical Python RSS column is an after-call reading, not a continuously sampled peak. Short-lived peaks can also be missed by Rust's 10 ms sampler.
- Python DB postprocess time is not split from detection by the current RapidOCR timing surface.
- Batch-size tuning is not covered. The current numbers exercise the configured cls/rec batch paths for a single image.
- The benchmark script can write Markdown records with `--out`; curated baseline updates should stay in this file.

## RSS Measurement Audit (2026-11-21)

The previous comparison combined a measurement bias with different ONNX Runtime settings. The Rust snapshot at `a8d48ab` used the default CPU arena allocator, while Python disabled it. Rust gained an explicit arena switch defaulting to `false` in `e80793c`; the current code also defaults to one intra-op thread, one inter-op thread, sequential execution, and disabled memory patterns for dynamic OCR inputs. The July RSS rows do not describe the current resource policy.

The corrected benchmark runs each implementation in a fresh child process, samples both continuously from startup through inference, and warms up both once before five measured calls. On Windows it also records the largest observed OS peak working set (`peak_wset`), which retains peaks that occur between RSS samples. The parent process and Cargo build are excluded. Python's old after-call reading is retained separately as a diagnostic.

This audit used Rust HEAD `5c73414` plus the benchmark measurement changes, Python RapidOCR HEAD `c18432567fca72c21d2f36dd800c0f4a5c44d01d`, Python 3.12.10, Rust 1.100.0, and Windows 11 build 26320. Both used **ONNX Runtime 1.26.0 from the same Python wheel** and the same local ONNX assets: Rust loaded its `onnxruntime.dll`, while Python used `onnxruntime_pybind11_state.pyd`. This aligns the runtime release/distribution, not the exact loaded native module. The input was `python/tests/test_files/ch_en_num.jpg` (323 by 430 pixels), below the Rust input resource limits. Both used batch size six, one intra-op thread, one inter-op thread, sequential execution, and disabled CPU arenas and memory patterns.

### Current PP-OCRv6 Small

| use_cls | runner | mean_ms | sampled peak RSS (MiB) | observed OS peak RSS (MiB) | sampled peak Private Bytes (MiB) | Python after-call RSS (MiB) | output lines |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true | `rust_cli_hot` | 1026.781 | 160.3 | 165.6 | 150.4 | | 18 |
| true | `python_hot` | 889.382 | 267.8 | 280.8 | 720.3 | 194.5 | 18 |
| false | `rust_cli_hot` | 859.398 | 162.0 | 163.7 | 147.6 | | 18 |
| false | `python_hot` | 865.056 | 255.7 | 278.9 | 718.3 | 194.3 | 18 |

The current pipeline did **not** reproduce higher Rust peak RSS in this local run. Python's after-call RSS was about 194 MiB even though the OS observed a peak near 280 MiB. Private Bytes measures committed private memory, not resident memory; it must not be interpreted as RSS or live tensor size. Matching output line counts is a workload sanity check, not a complete OCR correctness comparison. These latency results are from a different snapshot, runtime, OS, and model family than July and should not be used to infer a latency regression or speedup.

A separate release build with the **default bundled Rust runtime**, without `load-dynamic` or `--ort-dylib`, also returned 18 lines: sampled peak RSS was 161.8 MiB, observed OS peak RSS was 165.2 MiB, sampled Private Bytes peaked at 149.1 MiB, and mean time was 893.549 ms. The lower Rust peak therefore did not depend on choosing the Python wheel's runtime DLL.

Reproduce the current comparison after preparing the default model cache and installing the Python reference dependencies and `psutil` (the script requires Python 3.11+):

```powershell
$env:RAPIDOCR_PYTHON_REPO = "D:\projects\RapidOCR"
$rapidocrOrtDll = python -c "import pathlib, onnxruntime; print(pathlib.Path(onnxruntime.__file__).parent / 'capi' / 'onnxruntime.dll')"
python .\tools\bench_e2e.py --ort-dylib $rapidocrOrtDll --repeat 5 --out target\benchmark_rss_cls.md
python .\tools\bench_e2e.py --ort-dylib $rapidocrOrtDll --repeat 5 --no-cls --out target\benchmark_rss_no_cls.md
python .\tools\bench_e2e.py --repeat 5 --skip-python --out target\benchmark_rss_default_runtime.md
```

### CPU Arena Ablation on PP-OCRv5 Mobile

For a separate controlled arena check, both sides used the PP-OCRv5 mobile detector/recognizer and the older `ch_ppocr_mobile_v2.0_cls_mobile.onnx` classifier with input shape `[3, 48, 192]`. This matches the locally available July-era classifier, rather than the newer classifier in the current v5 model registry. The two Rust configs differed only in `enable_cpu_mem_arena`.

| CPU arena | runner | mean_ms | sampled peak RSS (MiB) | observed OS peak RSS (MiB) | Python after-call RSS (MiB) | output lines |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| false | `rust_cli_hot` | 1160.589 | 154.7 | 158.5 | | 18 |
| false | `python_hot` | 911.059 | 253.0 | 269.7 | 185.5 | 17 |
| true | `rust_cli_hot` | 687.523 | 270.1 | 270.1 | | 18 |
| true | `python_hot` | 632.333 | 397.7 | 415.7 | 388.7 | 17 |

The arena retained more memory and reduced allocation overhead in this sample: Rust's observed OS peak increased from 158.5 to 270.1 MiB. This verifies a real memory/latency tradeoff, but does not isolate every cause of the original July gap. Python and Rust returned different line counts for these v5 models, so the v5 rows are not a strict latency parity benchmark. The Python CPU provider's arena growth policy can also differ from Rust's when arenas are enabled.

### Remaining Memory Work

- Keep the default CPU arena disabled when memory is the priority. Enabling it may improve repeated-inference latency while retaining intermediate buffers in each model session.
- `OnnxSession::run_f32` still copies the entire ONNX output with `view.to_owned()`. Recognition subsequently decodes by view, but the session-level copy remains. Holding the native output tensor while decoding could remove this extra allocation; its size scales as `batch * timesteps * classes * sizeof(f32)`.
- Smaller recognition batches and narrower padded inputs can reduce the tensor/intermediate footprint, with a throughput tradeoff. Reducing detector input resolution also reduces activations but requires an OCR accuracy check.
- A high process RSS or high-water mark alone does not establish a leak. Investigate steady-state RSS across repeated requests separately from transient peaks and allocator caches.

## Tokio Convenience Overhead (0.2.1 Candidate)

Recorded on 2026-07-12 with five local images, two measured passes, two warm-up images, PP-OCRv5
Chinese mobile detection and recognition, classification disabled, two intra-op threads, one
inter-op thread, and the CPU memory arena disabled. Image contents and OCR text were not included
in the benchmark output.

| metric | synchronous `RapidOcr` | `TokioRapidOcr` | Tokio delta |
| --- | ---: | ---: | ---: |
| completed inferences | 10/10 | 10/10 | 0 |
| failures | 0 | 0 | 0 |
| mean elapsed | 1751.633 ms | 1742.985 ms | -0.49% |
| median elapsed | 1404.256 ms | 1399.381 ms | -0.35% |
| maximum elapsed | 3980.149 ms | 3937.445 ms | -1.07% |
| post-warmup RSS baseline | 153.5 MiB | 154.4 MiB | +0.9 MiB |
| measured RSS peak | 333.6 MiB | 334.0 MiB | +0.35 MiB |
| measured Private Bytes peak | 319.4 MiB | 318.1 MiB | -1.3 MiB |
| peak threads | 22 | 25 | +3 |
| peak handles | 137 | 148 | +11 |
| output lines | 822 | 822 | 0 |

The sample is too small to claim a speedup. It shows that the dedicated worker, bounded Tokio
channel, and oneshot response add no measurable latency or memory regression at OCR-scale request
durations. The expected persistent cost is three additional threads and eleven handles in this
Windows run.
