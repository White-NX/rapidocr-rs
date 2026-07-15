# rapidocr-rs

Rust ONNX Runtime OCR core for RapidOCR-style detection, optional text-line orientation classification, and text recognition pipelines.

## Introduction

`rapidocr-rs` is a Rust workspace for running RapidOCR model pipelines through ONNX Runtime. It currently provides:

- `rapidocr-core`: library API, model registry, TOML configuration, image preprocessing, OCR pipeline, and model cache handling.
- `rapidocr-cli`: workspace command-line wrapper for local validation and benchmarking.

The implementation is focused on the Rust core. It is not a full replacement for the Python RapidOCR package, and it does not include Python bindings, visualization parity, training, or non-ONNX inference backends.

## Project Origin

Building upon the existing ONNX models from RapidOCR, we developed this project using graphics algorithms that closely replicate the original functionality, all while avoiding dependencies on `cv2` (which can be difficult to compile). This enables developers to implement efficient OCR inference without relying on Python or complex libraries.

## Status

The workspace can run `det -> optional cls -> rec` with registered ONNX model sets. The detector postprocess is an OpenCV-free Rust implementation that approximates Python RapidOCR's `DBPostProcess`; parity gaps are tracked separately in [parity-gaps.md](parity-gaps.md).

## Build

```powershell
cargo build
cargo test
```

The default test path does not require downloaded ONNX models or an external Python RapidOCR checkout. Ignored parity and benchmark tests have extra setup described in [docs/development.md](docs/development.md).

## Quick Start

Run OCR from the CLI. Missing registered model assets are downloaded into `models` by default:

```powershell
cargo run -p rapidocr-cli -- --image path\to\image.png --model-dir models
```

Disable the direction classifier:

```powershell
cargo run -p rapidocr-cli -- --image path\to\image.png --model-dir models --no-cls
```

Run recognition on the whole image without detection:

```powershell
cargo run -p rapidocr-cli -- --image path\to\crop.png --model-dir models --no-det
```

Generate a TOML config:

```powershell
cargo run -p rapidocr-cli -- --write-default-config config\ppocrv6-small.toml --model-dir models
```

Use the library API:

```rust
use rapidocr_core::{
    config::{InferenceOptions, PipelineConfig},
    model::{model_set_by_name, ModelCache, ModelDownloadMode},
    RapidOcr,
};

fn main() -> anyhow::Result<()> {
    let model_set = model_set_by_name("ppocrv6-small").unwrap();
    let cache = ModelCache::new("models");
    cache.ensure_model_set_for_pipeline(
        model_set,
        PipelineConfig::without_cls(),
        ModelDownloadMode::Missing,
    )?;

    let cfg = cache
        .config_for(model_set)
        .with_pipeline(PipelineConfig::without_cls())
        .with_inference_options(InferenceOptions {
            intra_threads: 1,
            inter_threads: 1,
            parallel_execution: false,
            ..Default::default()
        });
    let mut ocr = RapidOcr::from_config(cfg)?;
    let output = ocr.run_path("path/to/image.png")?;

    for line in output.lines {
        println!("{:.5}\t{}", line.score, line.text);
    }

    Ok(())
}
```

The runnable example is in [crates/rapidocr-core/examples/library_usage.rs](crates/rapidocr-core/examples/library_usage.rs):

```powershell
cargo run -p rapidocr-core --example library_usage -- path\to\image.png
```

CLI examples are collected in [examples/cli_usage.ps1](examples/cli_usage.ps1).

### Detector input limits

Detection uses dynamic input shapes. A thin image can therefore request a much larger tensor when
the detector expands its minimum side. The default configuration keeps this internal amplification
bounded without changing ordinary desktop screenshots:

```toml
[det.input_limits]
max_side_len = 4096
max_pixels = 4194304
overflow_behavior = "downscale"
```

`downscale` preserves aspect ratio and fits the detector input into both limits. Applications that
must not silently reduce detection resolution can use `reject` and handle the returned error, for
example by tiling the image. Applications that deliberately accept the memory and latency risk can
use `allow`, which ignores both limits.

Rust callers can make the opt-out especially explicit:

```rust
use rapidocr_core::config::DetInputLimits;

let mut cfg = rapidocr_core::config::RapidOcrConfig::ppocr_v6_small("models");
cfg.det.as_mut().unwrap().input_limits = DetInputLimits::unrestricted();
```

Older TOML files without `[det.input_limits]` remain compatible and receive the safe defaults.

## Model Assets

The Rust crates do not bundle ONNX models or dictionaries. Model files are large, model choice depends on language and deployment requirements, and applications should make model distribution explicit.

Projects using `rapidocr-core` can handle models in one of three ways:

- Let `ModelCache::ensure_model_set` or `ModelCache::ensure_model_set_for_pipeline` download registered assets at install time, first run, or startup.
- Pre-populate a model directory in CI, Docker images, installers, or release archives, then use `ModelDownloadMode::Never` or CLI `--no-download`.
- Provide explicit `model_path` and `dict_path` values in a TOML config when models are managed by the application or an internal artifact store.

The default local model directory is `models`, which is ignored by git.

Model downloading is enabled by the default `model-download` Cargo feature. Applications that
pre-populate model files can avoid the blocking `reqwest` dependency with
`rapidocr-core = { version = "0.2.2", default-features = false }`.

On Windows, enable DirectML inference on a DirectX 12-capable GPU with:

```toml
rapidocr-core = { version = "0.2.2", features = ["directml"] }
```

Select `ExecutionProvider::DirectMl` in `InferenceOptions` to use it. DirectML initialization is
fail-fast, requires `parallel_execution = false`, and unsupported operators may still fall back to
the CPU provider. In TOML, set `inference.execution_provider = "direct-ml"`.

The workspace CLI forwards the feature and exposes an explicit flag:

```powershell
cargo run -p rapidocr-cli --features directml -- --directml --image path\to\image.png
```

## Cancellation and Tokio

Synchronous callers can cooperatively cancel a complete OCR pipeline with
`OcrCancellationToken` and `RapidOcr::run_image_cancellable` or
`RapidOcr::run_path_cancellable`. Active ONNX Runtime calls receive a real
`RunOptions::terminate` signal, while preprocessing, postprocessing, crops, and
recognition batches check the same token between bounded work units.

Enable the optional Tokio convenience layer with:

```toml
rapidocr-core = { version = "0.2.2", features = ["tokio"] }
```

`TokioRapidOcr` owns a dedicated OCR worker thread and a bounded request queue,
so stateful sessions are never used concurrently and Tokio executor threads are
not blocked by inference. `OcrTask` cancels on drop and provides
`cancel_and_wait` and cooperative `timeout` operations. A timeout requests
termination and then waits for cleanup; it is not a hard-real-time deadline.
Call `shutdown().await` to wait for deterministic worker cleanup. Dropping the
service requests cancellation but does not join the worker thread.

```powershell
cargo run -p rapidocr-core --features tokio --example tokio_usage -- path\to\image.png
```

## Supported Model Sets

- `ppocrv6-small` (default)
- `ppocrv6-tiny`
- `ppocrv6-medium`
- `ppocrv4-en-mobile`
- `ppocrv5-ch-mobile`
- `ppocrv5-en-mobile`
- `ppocrv5-ch-server`

Select a model set from the CLI:

```powershell
cargo run -p rapidocr-cli -- --model-set ppocrv5-en-mobile --image path\to\image.png --model-dir models
```

## Development

Developer workflows, parity fixture generation, ignored test gates, local config, and benchmark commands are documented in [docs/development.md](docs/development.md).

Additional project notes:

- [parity-gaps.md](parity-gaps.md) tracks known behavior differences from Python RapidOCR.
- [benchmark-baseline.md](benchmark-baseline.md) records the curated benchmark baseline.

## Acknowledgements

This project follows the RapidOCR model and pipeline conventions and uses ONNX Runtime through the Rust `ort` crate.

## License

Apache-2.0.
