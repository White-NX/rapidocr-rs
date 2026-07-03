# rapidocr-core

Rust OCR core for RapidOCR-style ONNX pipelines.

The crate provides:

- `RapidOcr` library API for `det -> optional cls -> rec`.
- `RapidOcrConfig` TOML-compatible configuration.
- Registered ONNX model sets through `model_set_by_name` and `available_model_sets`.
- Explicit model cache/download handling through `ModelCache` and `ModelDownloadMode`.

Models are not bundled in the crate package. Applications should either let `ModelCache` download registered model assets, pre-populate a model directory during deployment, or provide explicit model paths in configuration.

See the workspace `README.md` for parity, benchmark, and CLI workflows.
