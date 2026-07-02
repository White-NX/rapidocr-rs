# rapidocr-rs Roadmap

This roadmap tracks the Rust migration of RapidOCR. The current direction is not a full one-shot replacement of the Python package, but an incremental Rust core that first proves and stabilizes the default ONNX OCR pipeline.

## Current Status

`rapidocr-rs` is a runnable Rust MVP for the default OCR flow:

- ONNX Runtime inference through the `ort` crate.
- Pure Rust `det -> optional cls -> rec` pipeline.
- Pure Rust DB postprocess approximation without the OpenCV crate.
- Optional cls stage with `--no-cls` support.
- CLI for local validation.
- DBPostProcess parity fixtures.
- End-to-end golden/parity tests with output metrics and per-fixture tolerances for documented geometry gaps.
- Structured public `RapidOcrConfig` with `[pipeline]` det/cls/rec switches.
- Explicit default model metadata and `ModelCache` download/cache API.
- Initial end-to-end benchmark baseline against Python ONNX Runtime.

The project is currently usable for validating the default PPOCRv6 det/rec + PPOCRv4 cls ONNX pipeline, but it is not yet a complete Rust replacement for RapidOCR.

## Repository Direction

`rapidocr-rs` now lives as a standalone repository under `D:\projects\rapidocr-rs`.

The Rust crate does not rely on being nested inside the Python RapidOCR checkout. The Python repository is an optional external parity source, discovered through an explicit path such as `RAPIDOCR_PYTHON_REPO`, a tool `--python-repo` argument, or ignored local test config.

## Phase 1: Stabilize The Default ONNX Pipeline

Goal: make the default Rust OCR pipeline predictable, tested, and easy to run.

Status: mostly complete.

Scope:

- Detection model loading and preprocessing.
- Pure Rust DB postprocess.
- Text crop and perspective transform.
- Optional direction classification.
- Recognition model loading and CTC decode.
- CLI smoke path.
- Golden/parity tests for DB postprocess and e2e output.

Completion criteria:

- `cargo check` passes.
- `cargo test` passes.
- CLI can run on existing `python/tests/test_files` images.
- E2E parity metrics are recorded and enforced.
- `--no-cls` fully removes cls from the runtime pipeline.

Remaining work:

- Add more e2e fixture images, especially noisy, small-text, transparent, and additional non-Chinese cases.
- Split benchmark numbers by model load, det, cls, rec, postprocess, and memory usage.
- Continue documenting known parity gaps as new candidate fixtures are tested.

## Phase 2: Public API And Configuration

Goal: turn the MVP into a stable Rust library interface rather than a hard-coded prototype.

Priority: high.

Status: complete.

Tasks:

- Done: define a public `RapidOcr` API suitable for crate users, including `from_config`, `config`, `pipeline`, `run_path`, and `run_image`.
- Done: add structured TOML config loading with `[pipeline]` instead of only `RapidOcrConfig::ppocr_v6_small`.
- Done: support configurable model paths for det, cls, rec, and dict.
- Done: support `use_det`, `use_cls`, and `use_rec` style switches where they make sense.
- Done: make model download/cache behavior explicit and testable through `ModelSetSpec`, `ModelCache`, and `ModelDownloadMode`.
- Done: improve error messages for missing models, invalid images, invalid dicts, and ONNX load failures.
- Done: output structs derive `Serialize`/`Deserialize`.

Completion criteria:

- Users can configure the default pipeline without editing Rust code.
- CLI and library use the same config model.
- Missing assets produce actionable errors.
- Existing e2e tests still pass.

## Phase 3: Standalone Repository Split

Goal: make `rapidocr-rs` work as an independent repository while preserving parity workflows against the Python RapidOCR repo.

Priority: high.

Status: complete.

Tasks:

- Done: move the directory as an independent git repository under `D:\projects\rapidocr-rs`.
- Done: replace parent-directory assumptions in tests with an explicit Python RapidOCR repo path.
- Done: add helpers for locating Python parity assets:
  - `RAPIDOCR_PYTHON_REPO`
  - tool `--python-repo`
  - ignored `config/local.toml`
- Done: update fixture export tools so they can run from the standalone Rust repo.
- Done: update benchmark scripts so Rust and Python paths are explicit.
- Done: update README commands that previously used `..\python\tests\test_files`.
- Done: keep parity fixtures storing source image paths relative to the Python repo; test images are not copied into this repo.
- Done: keep generated models, local caches, and benchmark outputs out of git.

Completion criteria:

- `cargo check` passes from the standalone repo.
- `cargo test` passes from the standalone repo when models and the Python repo path are available.
- Tests that require Python parity assets fail with an actionable message when the Python repo path is missing.
- README commands work after cloning only `rapidocr-rs`, with any external Python repo dependency stated explicitly.
- The parent RapidOCR repo no longer needs to contain `rapidocr-rs`.

## Phase 4: Parity Expansion

Goal: broaden confidence beyond the current small fixture set.

Priority: high.

Status: in progress.

Current strict e2e coverage:

- `ch_en_num.jpg` and `text_det.jpg` with cls enabled, cls disabled, and detection-only.
- `check_return_word_len.jpeg` with cls enabled, cls disabled, and detection-only as dense-text checks with documented local text tolerance.
- `arabic.png`, `cyrillic.png`, `devanagari.jpg`, `japan.jpg`, and `korean.jpg` as cross-language detection-only geometry checks.
- `ta.png`, `th_rec.jpg`, `te.png`, and `eslav.jpg` as additional script/layout detection-only geometry checks.
- `te.png` with cls enabled and disabled as a default-model full-pipeline parity check.
- `eslav.jpg` with cls enabled and disabled as a full-pipeline parity check with documented local score tolerance.
- `en.jpg` and `latin.jpg` with cls enabled and disabled.
- `empty_black.jpg` and `short.png` with cls enabled and disabled.
- `black_font_color_transparent.png` with cls enabled and disabled for transparent-background handling.
- `white_font_color_transparent.png` as a detection-only geometry check with documented local corner-drift tolerance.
- `img_exif_orientation.jpg` with cls enabled and disabled for EXIF orientation handling.
- `ch_doc_server.png` with cls enabled and detection-only for tiny edge text.
- `test_letterbox_like.jpg` and `test_without_det.jpg` with cls enabled and disabled.
- `return_word_debug.jpg` with cls enabled for slanted text and digit-string recognition.
- `text_vertical_words.png` with cls enabled and disabled.
- `issue_170.png` with cls enabled and disabled; this fixture uses a documented local corner-drift tolerance.
- `en_rec.jpg`, `el_rec.jpg`, and `devanagari_rec.png` as recognition-crop detection-only geometry checks.
- `en_rec.jpg` as a recognition-only long English line cls/no-cls check.
- `el_rec.jpg` as a recognition-only Greek line cls/no-cls check.
- `devanagari_rec.png` as a recognition-only no-cls check.
- `text_rec.jpg` as a recognition-only normal-crop cls/no-cls check.
- `text_cls.jpg` as a recognition-only 180-degree cls/no-cls check.
- `text_cls.jpg` as a Rust golden for the cls/no-cls pipeline switch.

Current DBPostProcess coverage:

- `ch_en_num.jpg`
- `text_det.jpg`
- `en.jpg`
- `test_letterbox_like.jpg`
- `test_without_det.jpg`
- `text_vertical_words.png`
- `latin.jpg`
- `img_exif_orientation.jpg`
- `empty_black.jpg`
- `issue_170.png`
- `short.png`
- `return_word_debug.jpg`
- `en_rec.jpg`
- `el_rec.jpg`
- `devanagari_rec.png`
- `black_font_color_transparent.png`
- `white_font_color_transparent.png`
- `ch_doc_server.png`
- `check_return_word_len.jpeg`
- `arabic.png`
- `cyrillic.png`
- `devanagari.jpg`
- `japan.jpg`
- `korean.jpg`
- `ta.png`
- `th_rec.jpg`
- `te.png`
- `eslav.jpg`

Tasks:

- In progress: expand e2e golden fixtures across existing Python test images.
- Done for strict e2e tests: track metrics per image:
  - line count
  - matched count
  - exact text ratio
  - character accuracy
  - score drift
  - center drift
  - corner drift
- Done for recognition-only e2e crops: add tests for cls-specific behavior:
  - normal text should not be rotated
  - 180-degree text should be rotated
  - `--no-cls` should preserve unrotated crops
  - Greek rec crop matches with cls enabled and disabled
  - Devanagari rec crop matches with cls disabled while cls-enabled behavior remains documented
- Done for detection-only e2e fixtures:
  - `ch_en_num.jpg`
  - `text_det.jpg`
  - `check_return_word_len.jpeg`
  - `arabic.png`
  - `cyrillic.png`
  - `devanagari.jpg`
  - `ch_doc_server.png`
  - `japan.jpg`
  - `korean.jpg`
  - `ta.png`
  - `th_rec.jpg`
  - `te.png`
  - `eslav.jpg`
  - `en_rec.jpg`
  - `el_rec.jpg`
  - `devanagari_rec.png`
  - count and geometry are gated without recognition text/score checks
- Done: full-pipeline `eslav.jpg` cls/no-cls fixtures with a local score-drift gate while text and geometry remain strict.
- In progress: add regression fixtures for DBPostProcess edge cases:
  - Done: empty images
  - Partial: dense small text; `ch_doc_server.png` is an e2e cls, detection-only, and DBPostProcess gate, and `check_return_word_len.jpeg` is now a cls/no-cls e2e, detection-only, and DBPostProcess gate with local text tolerance. The remaining dense/tiny-text gap is `ch_doc_server.png` no-cls text drift.
  - Done: vertical text
  - Done: slanted text
  - Done: Latin and EXIF-oriented DBPostProcess layout fixtures.
  - Deferred: low contrast text; the current Python `test_files` set has no dedicated low-contrast source image, so the closest strict gate is the near-threshold white transparent text case.
  - Done: representative cross-language DBPostProcess fixtures for Arabic, Cyrillic, Devanagari, Japanese, and Korean text layout.
  - Done: additional script/layout DBPostProcess fixtures for Tamil, Thai crop, Telugu, and Eslav images.
  - Done: recognition-crop DBPostProcess fixtures for English, Greek, and Devanagari crops.
  - Partial: transparent images; `black_font_color_transparent.png` is both a DBPostProcess and e2e gate, and `white_font_color_transparent.png` is a DBPostProcess and detection-only e2e gate with documented local geometry tolerance while its full e2e low-confidence recognition line remains a gap.

Completion criteria:

- E2E tests cover a representative subset of `python/tests/test_files`.
- Metrics are strict enough to catch real regressions and loose enough to tolerate known Rust/Python implementation differences.
- Current parity gaps are listed in README or a dedicated notes file.

## Phase 5: Performance And Memory

Goal: understand whether Rust improves deployability and runtime characteristics.

Priority: medium.

Status: started.

Tasks:

- Done: add repeatable benchmark commands through `tools/bench_e2e.py`.
- Done: record an initial hot-loop baseline in `benchmark-baseline.md`.
- Compare Rust `ort` vs Python `onnxruntime` on:
  - Todo: model load time
  - Todo: det latency
  - Todo: cls latency
  - Todo: rec latency
  - Started: end-to-end latency
  - Todo: memory usage
- Measure postprocess cost separately from ONNX inference.
- Avoid unnecessary image and tensor copies.
- Consider batch behavior for rec and cls.

Completion criteria:

- Benchmark results are documented.
- Major bottlenecks are identified.
- Obvious avoidable copies are removed or justified.

## Phase 6: Model Matrix

Goal: move beyond one hard-coded default model set.

Priority: medium.

Tasks:

- Support additional ONNX model families used by RapidOCR:
  - PP-OCRv4
  - PP-OCRv5
  - PP-OCRv6
- Support mobile/server variants where practical.
- Support language-specific rec dictionaries.
- Add model metadata definitions with URL, sha256, input shape, task type, and dict path.
- Add tests for at least one non-default recognition language.

Completion criteria:

- Model selection is data-driven.
- Adding a model does not require editing pipeline logic.
- At least two model families are validated by smoke/e2e tests.

## Phase 7: Packaging And Integration

Goal: make the Rust core usable outside the repository.

Priority: medium.

Tasks:

- Clean up crate metadata.
- Add CI for `cargo fmt`, `cargo check`, and `cargo test`.
- Decide release policy for bundled vs downloaded models.
- Add examples for library and CLI usage.
- Consider Python binding only after the Rust API is stable.
- Consider C ABI or other bindings only after core behavior is stable.

Completion criteria:

- The crate can be built and tested from a clean checkout.
- CLI usage is documented.
- Library usage is documented.
- Packaging does not require OpenCV.

## Non-Goals For Now

- Reimplement every RapidOCR backend immediately.
- Add the OpenCV crate.
- Add Paddle, PyTorch, OpenVINO, TensorRT, or MNN backends before ONNX behavior is stable.
- Claim full Python RapidOCR replacement before model/config/parity coverage exists.
- Optimize before benchmark evidence exists.

## Recommended Next Step

The next best step is parity/performance work from the standalone repository:

1. Test additional candidate fixtures before adding them to strict gates, especially noisy, small-text, transparent, EXIF-orientation, and additional language cases.
2. Extend focused cls behavior tests beyond recognition-only crops if detector-produced crops reveal additional rotation edge cases.
3. Extend the benchmark baseline beyond hot-loop e2e latency into model load, stage-level latency, postprocess cost, and memory usage.
4. Keep `parity-gaps.md` updated with rejected or deferred candidate fixtures and their observed metrics.

This keeps the project moving from a stable default API and repository layout toward broader parity and measured performance work without prematurely expanding the backend/model matrix.
