# rapidocr-core

Rust OCR core for RapidOCR-style ONNX pipelines.

The crate provides:

- `RapidOcr` library API for `det -> optional cls -> rec`.
- `RapidOcrConfig` TOML-compatible configuration.
- `OcrResult<T>` / `OcrError` classify configuration, model, runtime, input, resource-limit, and cancellation failures.
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

`types::OcrLine::score` always means recognition confidence. Detection-only lines
keep empty text and `score = 0.0`, but expose their DB box confidence through
`detection_score: Option<f32>`. Classification returns `rotation: Option<OcrRotation>`
with the winning `label`, classification `score`, and `applied_degrees` (`0` or `180`).
A low-confidence `"180"` label is retained with zero applied rotation. The angle
describes the classifier's additional crop rotation, excluding EXIF and tall-box
crop orientation; output box coordinates are unaffected. Disabled stages have `None`
metadata. `OcrLine::new` and `detection_only` start without metadata; use
`with_detection_score` and `with_rotation` to attach it when constructing results.
Serde accepts older JSON lines without these fields and serializes absent detection/rotation metadata as `null`.

## Optional token selection cells

`RecConfig::with_return_tokens(true)` (or `rec.return_tokens = true` in TOML)
adds `OcrLine::text_alignment: Option<OcrTextAlignment>`. Its default is disabled,
and `None` is **omitted** from JSON. Detection-only lines always have `None`.
Recognition-only lines use the full caller image as their crop; `run_path` applies
EXIF orientation before establishing that coordinate space.

`OcrTextAlignment` exposes `input_width` (padded batch width), `valid_width`
(actual resized crop width), `time_steps` (`T` in `[N, T, C]`), `tokens`, and
the optional per-line `mapping` described below.
The full probability tensor stays internal. Each `OcrToken` returns a half-open
UTF-8 `text_range`, `score`, half-open `timestep_range`, normalized `x_center`
and `x_range`, and a source-image `bbox`. Use `&line.text[token.text_range.clone()]`
to recover a token. Blank-separated repeated labels remain separate tokens;
spaces, combining characters and multi-code-point dictionary entries keep their
complete byte ranges. Consumers that use UTF-16 offsets must convert explicitly.

For a greedy CTC run `[s, e)`, the center is computed as:

```text
x_center = clamp(((s + e) / 2) * input_width / (time_steps * valid_width), 0, 1)
```

The division uses real preprocessing widths, including integer resize rounding
and any truncation to the padded batch width. Interior selection boundaries are
midpoints of neighboring centers; outer cells extend to `0` and `1`. The token
score remains the first retained timestep confidence, preserving the existing
mean recognition score. It does not estimate position accuracy.

`x_range` follows the recognition crop **after** the tall-crop 270-degree rotation
and any classifier 180-degree rotation. To obtain `bbox`, the pipeline maps cell
corners across crop pixel centers (`0..width-1`, `0..height-1`) through the exact
saved inverse crop transform and classifier rotation, then removes detection
padding, rescales, and clips to the caller image. The actual perspective transform
is saved before line-box clipping; edge-replicated and axis-aligned fallback crops
retain their own transforms. Corner order follows the recognition cell and may
therefore be rotated relative to source-image top-left order.

### Compact selection geometry

The opt-in OCR pipeline supplies `OcrTextAlignment::mapping: Option<OcrTextMapping>`
for every returned recognition line. This stores an **unclipped** normalized-to-source
transform once per line, so consumers can reconstruct cells without persisting every
token's four corners. `matrix: [[f32; 3]; 3]` is row-major and multiplies the column
vector `[u, v, 1]`, where `[0, 1]` spans the post-rotation recognition rectangle:

```text
h = matrix * [u, v, 1]^T
source_point = [h[0] / h[2], h[1] / h[2]]
```

The matrix includes pixel-center scaling, tall-crop and classifier rotations,
the perspective/fallback crop, replicated border, and detector resize/padding.
It is retained before source-image clipping; do not replace it with a transform
fitted to a clipped line or token quad. `source_size: [u32; 2]` stores source
`[width, height]`, after EXIF correction for `run_path`.

`mapping.project_point([u, v])` returns an unclipped point.
`mapping.selection_quad([a, b])` maps `(a, 0)`, `(b, 0)`, `(b, 1)`, `(a, 1)` in
recognition order and clips coordinates to `0..width-1`, `0..height-1`.
The helpers return `None` for invalid projections; cell reconstruction also rejects
empty source dimensions and intervals outside ordered `[0, 1]`. A matrix for a
one-pixel-wide/high crop can be singular and still project points successfully.

The coefficients use `f32`; retain their full values when storing a compact layout.
Small floating-point rounding differences can remain between reconstructed and
emitted token corners. `OcrTextAlignment::new` still constructs metadata without
a mapping; attach one with `with_mapping`. Serde accepts older alignment JSON with
no `mapping` field and omits this field when it is `None`.

These full-height cells are suitable for selection hit testing and highlights.
They are estimated CTC selection intervals rather than precise glyph boxes or
word segmentation. Alignment gaps, variable-width punctuation and OCR errors can
shift boundaries. Tokens emitted in right padding stay in the text and are clipped
to the crop edge, possibly producing zero-width cells. Enabling the option adds
per-token metadata and coordinate work, with no additional inference or model.
The default path does not allocate alignment vectors or store crop transforms.

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

All synchronous public fallible APIs return `Result<T, OcrError>`. Model errors carry
`ModelErrorKind` and an asset path; runtime errors carry `RuntimeErrorKind`; input-limit
errors carry the stage, resource, requested amount, and limit. Match variants or call
`error.kind()` instead of matching text. Original backend and I/O causes are retained
through `std::error::Error::source`. Tokio wraps these in `TokioOcrError::Ocr(OcrError)`.
`is_cancelled_error` now accepts `&OcrError`; these signature changes belong to `0.3.0`.
File loading checks decoded image bytes against `ImageReader`'s default 512 MiB budget
before pixel allocation and reports `InputLimitKind::ImageBytes`. Caller-loaded RGB
images passed to `run_image` use the caller's image-loading policy.

For deployment configuration, use `RapidOcrConfig::from_toml_file_with_options` with
`ConfigLoadOptions::new()`: strict parsing and model/dictionary paths relative to the
TOML file's directory. Unknown fields in any section and contradictory legacy/nested
pipeline flags are rejected. Legacy flags alone, equal aliases, and existing optional
defaults remain supported. `ModelPathBase` can instead preserve paths, snapshot the
working directory, or select an explicit directory. Resolved paths are absolute and
do not require assets to exist. Absolute paths are preserved.

The original `from_toml_file` / `from_toml_str` retain permissive parsing, legacy flag
precedence, and paths as written. `ConfigLoadOptions::compatible()` reproduces that
behavior. The `_strict` methods add strict parsing alone; `from_toml_str_with_base`
and `resolve_model_paths(base)` provide explicit resolution for string or existing configs.

See the workspace `README.md` for parity, benchmark, and CLI workflows.
