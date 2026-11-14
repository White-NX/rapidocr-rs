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

Rust 1.92 or newer is required. The `ort` and `ort-sys` dependencies are pinned
to `2.0.0-rc.15`, including the fixes for Cargo's Rust 1.100 output layout.

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

fn main() -> rapidocr_core::OcrResult<()> {
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
        .with_inference_options(
            InferenceOptions::new()
                .with_intra_threads(1)
                .with_inter_threads(1)
                .with_parallel_execution(false),
        );
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

Downloads are streamed to a sibling `<filename>.part` file while SHA-256 is
calculated. The part file is flushed and synchronized, then renamed into place
only after verification succeeds. Interrupted or invalid downloads do not leave
the final model filename behind. Cache preparation takes a cache-wide file lock
before downloading so cooperating processes do not race over the same part file.

Model downloading is enabled by the default `model-download` Cargo feature. Applications that
pre-populate model files can avoid the blocking `reqwest` dependency with
`rapidocr-core = { version = "0.3.0", default-features = false, features = ["download-binaries"] }`.

## Runtime Deployment

Model assets and the native ONNX Runtime library have separate Cargo features:

| Feature | Purpose |
| --- | --- |
| `model-download` | Download registered ONNX models and dictionaries at runtime. Enabled by default. |
| `download-binaries` | Download and prepare native ONNX Runtime binaries during the build. Enabled by default. |
| `load-dynamic` | Load an application-provided native library through `OcrRuntime::load_from` or `ORT_DYLIB_PATH`. |
| `directml` | Make the Windows DirectML provider available for explicit selection. |
| `tokio` | Enable the bounded worker service. |

Applications shipping their own models and native runtime can use:

```toml
rapidocr-core = { version = "0.3.0", default-features = false, features = ["load-dynamic", "tokio", "directml"] }
```

```rust
use rapidocr_core::{OcrRuntime, RapidOcr};

let runtime = OcrRuntime::load_from("runtime/onnxruntime.dll")?;
let config = rapidocr_core::config::RapidOcrConfig::ppocr_v6_small("models");
let mut ocr = RapidOcr::new_with_runtime(config, &runtime)?;
```

For Tokio, pass the same handle to `TokioRapidOcr::new_with_runtime(config, runtime).await`.
Initialize the native library before using other `ort` APIs. Repeated default constructors
share one process-wide environment, and changing its library path after initialization returns
an error. Failed library loads can be retried. A successfully initialized default environment
stays alive for the process lifetime; shutting down a worker releases its model sessions.

The crate explicitly selects ONNX Runtime API 1.24. Application-provided 1.24.x libraries
remain supported, while the default `ort` download currently supplies 1.31 binaries.
Cargo combines `ort` features across dependencies: if an application also depends on `ort`,
disable its defaults and select `api-24` to keep compatibility with a 1.24 runtime. To share
an application-owned environment, use `OcrRuntime::from_environment`; the interoperability
methods require the same pinned `ort` version.

Without `download-binaries` or `load-dynamic`, builds must provide their own linkable runtime
through `ORT_LIB_PATH`. DirectML deployments also need the matching provider dependencies,
including `DirectML.dll`.

On Windows, enable DirectML inference on a DirectX 12-capable GPU with:

```toml
rapidocr-core = { version = "0.3.0", features = ["directml"] }
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
rapidocr-core = { version = "0.3.0", features = ["tokio"] }
```

`TokioRapidOcr` owns a dedicated OCR worker thread and a bounded request queue,
so stateful sessions are never used concurrently and Tokio executor threads are
not blocked by inference. `OcrTask` cancels on drop and provides
`cancel_and_wait` and cooperative `timeout` operations. A timeout requests
termination and then waits for cleanup; it is not a hard-real-time deadline.
Call `shutdown().await` to wait for deterministic worker cleanup. Dropping the
service requests cancellation but does not join the worker thread.

Model preparation through `ModelCache` is synchronous. Run it before entering a Tokio
runtime or inside `tokio::task::spawn_blocking` when HTTP downloads may be needed.

## Model Contracts and Recognition Limits

Enabled models are checked for float32 NCHW input and stage-specific output shapes.
Recognition output classes must match the dictionary, including CTC blank and appended space;
classification classes must match the labels. Inference verifies the batch size before decoding.
Incompatible assets return errors instead of dropping unknown characters or indexing past crops.

Recognition pads each batch to its widest crop. These defaults bound that expansion before
allocating a tensor:

```toml
[rec.input_limits]
max_width = 4096
max_batch_pixels = 2097152
```

Exceeding a recognition limit returns an error. Applications can tile long text lines, reduce
`rec.batch_size`, raise the limits, or explicitly choose `RecInputLimits::unrestricted()`.
In TOML, use `"unlimited"` for an individual detector or recognition limit. An omitted limits
table receives the safe defaults. The limits cover input tensors, rather than total process memory.

```powershell
cargo run -p rapidocr-core --features tokio --example tokio_usage -- path\to\image.png
```

## Public Errors

Synchronous OCR, runtime, model-cache, and configuration APIs return `OcrResult<T>`
(`Result<T, OcrError>`). Match the public variants or call `error.kind()`; diagnostic
text is for humans and is not a matching contract.

| Error variant | Recovery information |
| --- | --- |
| `Model` | `ModelErrorKind` distinguishes missing assets, checksum failures, incompatible models/dictionaries, I/O, and downloads; `path` identifies the asset. |
| `Runtime` | `RuntimeErrorKind` distinguishes initialization, session/provider setup, execution, cancellation signaling, and worker failures. Inspect the original cause before retrying. |
| `InputLimit` | `stage`, `kind`, `requested`, and `limit` identify the resource; tile the input, reduce batching, or adjust the policy. |
| `InvalidInput` | Repair the image or path before retrying. |
| `Config` | `ConfigErrorKind` distinguishes parsing, unknown fields, conflicting flags, validation, I/O, serialization, and path resolution. |
| `Cancelled` | Stop the request; `is_cancelled_error` accepts `&OcrError`. |

Original ONNX, image-decoder, network, and I/O errors remain accessible through
`std::error::Error::source`, including the ONNX error code. Tokio exposes the same
typed error in `TokioOcrError::Ocr(OcrError)` and retains its explicit queue,
cancellation, timeout, and worker states. These return-type changes are part of the
`0.3.0` compatibility boundary; the core no longer depends on `anyhow`.

File loading checks decoded image bytes before pixel allocation using `ImageReader`'s
default 512 MiB allocation limit. These failures use `InputLimitKind::ImageBytes`;
applications with their own image-loading policy can supply an RGB image to `run_image`.

## Configuration Parsing and Path Bases

Use an explicit file-loading policy for deployments:

```rust
use rapidocr_core::config::{ConfigLoadOptions, RapidOcrConfig};

let config = RapidOcrConfig::from_toml_file_with_options(
    "settings/ocr.toml",
    &ConfigLoadOptions::new(),
)?;
```

`ConfigLoadOptions::new()` selects strict parsing and the configuration file's
directory as the model/dictionary path base. Strict parsing rejects unknown fields
at every level, including disabled stages. Legacy top-level `use_det`, `use_cls`,
and `use_rec` remain accepted; a flag also explicitly present in `[pipeline]` must
have the same value. Missing optional settings keep their existing defaults.

| Entry point | Parsing | Model/dictionary paths |
| --- | --- | --- |
| `from_toml_str`, `from_toml_file` | Compatible: unknown fields ignored, legacy flags override `[pipeline]` | As written; relative paths follow the working directory at use time |
| `from_toml_str_strict`, `from_toml_file_strict` | Strict | As written |
| `from_toml_str_with_base` | Strict | Absolute paths rooted at the supplied directory |
| `from_toml_file_with_options` | Selected by `ConfigParseMode` | Selected by `ModelPathBase`; defaults to the configuration directory |

`ModelPathBase` offers `AsProvided`, `WorkingDirectory`, `ConfigDirectory`, and
`Directory(PathBuf)`. Resolution snapshots relative paths as absolute paths without
requiring the assets to exist, so configuration can load before model preparation.
Absolute asset paths are preserved. Relative explicit base directories follow the
working directory at load time; `..` components are allowed. On Windows, drive-relative
(`C:models`) and root-relative (`\models`) paths require an explicit drive when resolving.
Call `resolve_model_paths(base)` to resolve an existing configuration; it updates all
configured stages atomically. `ConfigLoadOptions::compatible()` reproduces the original
file loader's complete parsing and path behavior.

The CLI parses strictly by default and keeps the working directory as its default
path base, preserving existing bundled configurations. Use `--config-compat` for
legacy unknown-field/override behavior and paths as written with the default base,
`--config-path-base config` for a base at the
TOML file's directory, or `--config-base-dir <directory>` for an explicit base.
With `--config-path-base config`, a TOML file in `config/` should use `../models/...`
to refer to the workspace model directory.

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

- [DESIGN_REVIEW.md](DESIGN_REVIEW.md) records the design audit, runtime integration changes, and consumer migration guidance.
- [RELEASE_NOTES.md](RELEASE_NOTES.md) records the `0.3.0` API and reliability changes.
- [parity-gaps.md](parity-gaps.md) tracks known behavior differences from Python RapidOCR.
- [benchmark-baseline.md](benchmark-baseline.md) records the curated benchmark baseline.

## Acknowledgements

This project follows the RapidOCR model and pipeline conventions and uses ONNX Runtime through the Rust `ort` crate.

## License

Apache-2.0.
