# rapidocr-cli

Command-line wrapper for validating `rapidocr-core` OCR pipelines.

This package is currently a workspace tool and is not published independently.

The binary is named `rapidocr` and can:

- Run OCR on an image.
- Select registered model sets with `--model-set`.
- Generate TOML configuration with `--write-default-config`.
- Disable pipeline stages with `--no-det`, `--no-cls`, and `--no-rec`.
- Emit benchmark timing JSON with `--benchmark-json`.

Models are not bundled in the crate package. Use `--model-dir` with the default download behavior, pass `--no-download` when models are pre-populated, or run from an explicit TOML config.

On Windows, pass `--features directml` at build time and `--directml` at runtime to use the
DirectML execution provider on a DirectX 12-capable GPU.

See the workspace `README.md` for full commands and parity workflows.
