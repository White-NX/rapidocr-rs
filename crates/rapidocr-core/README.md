# rapidocr-core

Rust OCR core for RapidOCR-style ONNX pipelines.

The crate provides:

- `RapidOcr` library API for `det -> optional cls -> rec`.
- `RapidOcrConfig` TOML-compatible configuration.
- `InferenceOptions` limits ONNX Runtime intra-op/inter-op threads and parallel execution.
- `OcrRuntime` initializes and shares one ONNX environment, with explicit application-owned runtime support.
- `VERSION` reports the compiled crate version for consumer diagnostics.
- `OcrCancellationToken` cooperatively cancels active ONNX runs and later pipeline stages.
- The optional `tokio` feature provides a dedicated-worker `TokioRapidOcr` service and cancel-on-drop tasks.
- Registered ONNX model sets through `model_set_by_name` and `available_model_sets`.
- Explicit model cache/download handling through `ModelCache` and `ModelDownloadMode`.

Public configuration, model-description, and OCR output types that may grow are
`#[non_exhaustive]`. Construct them through their `new` methods, builders, or
`Default` instead of downstream struct literals.

Models are not bundled in the crate package. Applications should either let `ModelCache` download registered model assets, pre-populate a model directory during deployment, or provide explicit model paths in configuration.

The default `model-download` feature enables the blocking HTTP downloader. Disable default
features and enable `download-binaries` when models are provisioned separately to avoid
depending on `reqwest`. For application-provided native libraries, enable `load-dynamic`
instead and initialize `OcrRuntime::load_from` before constructing a pipeline or worker.
Downloads stream to `<filename>.part`, compute SHA-256 during the write, synchronize the
part file, and rename it into place only after verification succeeds. A cache-wide
file lock serializes downloads from cooperating processes.

The optional Windows-only `directml` feature makes the DirectML execution provider available.
Select `ExecutionProvider::DirectMl` in `InferenceOptions` to use it. DirectML requires a DirectX
12-capable device and serial session execution.

The optional `tokio` feature adds a bounded, single-worker asynchronous adapter. Its timeout API
uses ONNX Runtime termination and waits for worker cleanup, but termination remains cooperative
rather than hard-real-time. Call `shutdown().await` for deterministic cleanup; dropping the adapter
requests cancellation but does not join its worker thread.

Rust 1.92+ is required. `ort` and `ort-sys` are pinned to `2.0.0-rc.15`; native API
1.24 is explicitly selected so application-provided 1.24.x runtimes remain usable.
`RapidOcr::new_with_runtime` and `TokioRapidOcr::new_with_runtime` accept a shared runtime.
Applications using `ort` directly can wrap their environment with `OcrRuntime::from_environment`.

Model input/output shapes and dictionary or label counts are validated. Recognition input
defaults to at most 4096 pixels wide and 2097152 padded pixels per batch. Oversized batches
return an error before tensor allocation; `RecInputLimits::unrestricted()` explicitly disables
these limits. TOML represents a disabled individual limit with `"unlimited"`.

See the workspace `README.md` for parity, benchmark, and CLI workflows.
