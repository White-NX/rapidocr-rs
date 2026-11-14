use std::path::PathBuf;

use rapidocr_core::config::{RapidOcrConfig, RecConfig};

pub fn model_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(filename)
}

pub fn recognition_config(filename: &str) -> RapidOcrConfig {
    RapidOcrConfig::new(
        None,
        None,
        Some(RecConfig::new(
            model_path(filename),
            model_path("one-token.txt"),
            [3, 48, 320],
            1,
        )),
    )
}
