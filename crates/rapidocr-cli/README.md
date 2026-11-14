# rapidocr-cli

Command-line wrapper for validating `rapidocr-core` OCR pipelines.

This package is currently a workspace tool and is not published independently.

The binary is named `rapidocr` and can:

- Run OCR on an image.
- Select registered model sets with `--model-set`.
- Generate TOML configuration with `--write-default-config`.
- Parse TOML strictly, with an explicit compatibility flag and model-path base selection.
- Disable pipeline stages with `--no-det`, `--no-cls`, and `--no-rec`.
- Emit benchmark timing JSON with `--benchmark-json`.

Each output row starts with recognition confidence, text, and box coordinates.
Enabled stages append labeled `det_score`, `cls_label`, `cls_score`, and `rotation`
values. In detection-only mode, text is empty and recognition confidence is `0.0`;
use `det_score` for detection confidence. `rotation` is the additional classifier
rotation actually applied to the crop (`0deg` or `180deg`), which can be `0deg`
even when `cls_label` is `"180"` if the classification threshold was not exceeded.

Models are not bundled in the crate package. Use `--model-dir` with the default download behavior, pass `--no-download` when models are pre-populated, or run from an explicit TOML config.

`--config` rejects unknown fields and contradictory legacy/nested pipeline flags by
default. `--config-compat` restores the original unknown-field tolerance and top-level
flag precedence. Legacy flags alone and equal aliases remain accepted in strict mode.
With the default path base, compatibility mode also keeps paths as supplied, including
empty paths in disabled stages. An explicit configuration-directory or custom base still
requests path resolution.

Relative model and dictionary paths follow the working directory by default, preserving
existing configurations. Use `--config-path-base config` for paths relative to the TOML
file's directory or `--config-base-dir <directory>` for an explicit base. Absolute paths
are preserved. For a TOML file in `config/`, use `../models/...` with the configuration
directory policy. Generated configs keep `--model-dir` paths as supplied.

On Windows, pass `--features directml` at build time and `--directml` at runtime to use the
DirectML execution provider on a DirectX 12-capable GPU.

See the workspace `README.md` for full commands and parity workflows.
