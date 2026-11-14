#[path = "common/temp_dir.rs"]
mod temp_dir;

use std::{error::Error, fs, path::PathBuf};

use rapidocr_core::{
    config::{ConfigLoadOptions, ConfigParseMode, ModelPathBase, PipelineConfig, RapidOcrConfig},
    ConfigErrorKind, OcrError, OcrErrorKind, OcrResult,
};
use temp_dir::TestDir;

fn config_text() -> String {
    RapidOcrConfig::ppocr_v6_small("models")
        .to_toml_string()
        .unwrap()
}

#[test]
fn strict_parsing_rejects_unknown_fields_in_every_section_including_disabled_stages() {
    for prefix in [
        "",
        "pipeline",
        "inference",
        "det",
        "cls",
        "rec",
        "det.input_limits",
        "rec.input_limits",
    ] {
        let config = RapidOcrConfig::ppocr_v6_small("models")
            .with_pipeline(PipelineConfig::recognition_only());
        let mut value: toml::Value = config.to_toml_string().unwrap().parse().unwrap();
        let mut section = &mut value;
        for component in prefix.split('.').filter(|component| !component.is_empty()) {
            section = section.get_mut(component).unwrap();
        }
        section
            .as_table_mut()
            .unwrap()
            .insert("typo".to_owned(), toml::Value::Boolean(true));
        let text = toml::to_string(&value).unwrap();
        let error = RapidOcrConfig::from_toml_str_strict(&text).unwrap_err();
        let expected_field = if prefix.is_empty() {
            "typo".to_owned()
        } else {
            format!("{prefix}.typo")
        };
        assert!(
            matches!(error, OcrError::Config { kind: ConfigErrorKind::UnknownField, field: Some(field), .. } if field == expected_field)
        );
        RapidOcrConfig::from_toml_str(&text).unwrap();
    }
}

#[test]
fn strict_flags_accept_legacy_only_and_equal_aliases_but_reject_conflicts() {
    let text = config_text();
    for name in ["use_det", "use_cls", "use_rec"] {
        let equal = format!("{name} = true\n{text}");
        RapidOcrConfig::from_toml_str_strict(&equal).unwrap();
        let conflict = format!("{name} = false\n{text}");
        let error = RapidOcrConfig::from_toml_str_strict(&conflict).unwrap_err();
        assert!(
            matches!(error, OcrError::Config { kind: ConfigErrorKind::ConflictingField, field: Some(field), .. } if field == format!("pipeline.{name}"))
        );
    }
    let mut value: toml::Value = text.parse().unwrap();
    let table = value.as_table_mut().unwrap();
    table.remove("pipeline");
    table.insert("use_det".into(), false.into());
    table.insert("use_cls".into(), false.into());
    table.insert("use_rec".into(), true.into());
    let legacy = toml::to_string(&value).unwrap();
    assert_eq!(
        RapidOcrConfig::from_toml_str_strict(&legacy)
            .unwrap()
            .pipeline,
        PipelineConfig::recognition_only()
    );

    let conflict = format!("use_det = false\nuse_cls = false\n{text}");
    assert_eq!(
        RapidOcrConfig::from_toml_str(&conflict).unwrap().pipeline,
        PipelineConfig::recognition_only()
    );
}

#[test]
fn parse_io_and_validation_errors_have_categories_and_retain_original_causes() {
    let result: OcrResult<RapidOcrConfig> = RapidOcrConfig::from_toml_str_strict("text_score = [");
    let error = result.unwrap_err();
    assert_eq!(error.kind(), OcrErrorKind::Configuration);
    assert!(matches!(
        &error,
        OcrError::Config {
            kind: ConfigErrorKind::Parse,
            ..
        }
    ));
    assert!(error.source().unwrap().is::<toml::de::Error>());

    let dir = TestDir::new("config-errors");
    let error = RapidOcrConfig::from_toml_file(dir.path().join("missing.toml")).unwrap_err();
    assert!(matches!(
        &error,
        OcrError::Config {
            kind: ConfigErrorKind::Io,
            ..
        }
    ));
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::NotFound
    );

    let mut config = RapidOcrConfig::ppocr_v6_small("models");
    config.inference.intra_threads = 0;
    assert!(matches!(
        config.validate().unwrap_err(),
        OcrError::Config {
            kind: ConfigErrorKind::Validation,
            ..
        }
    ));
}

#[test]
fn default_file_options_use_config_directory_and_resolve_disabled_assets_without_existence_checks()
{
    let dir = TestDir::new("config-relative");
    let path = dir.path().join("ocr.toml");
    let mut config =
        RapidOcrConfig::ppocr_v6_small("models").with_pipeline(PipelineConfig::recognition_only());
    let absolute_model = dir.path().join("elsewhere/rec.onnx");
    config.rec.as_mut().unwrap().model_path = absolute_model.clone();
    config.rec.as_mut().unwrap().dict_path = PathBuf::from("../shared/dict.txt");
    config.write_toml_file(&path).unwrap();

    let mut loaded =
        RapidOcrConfig::from_toml_file_with_options(&path, &ConfigLoadOptions::new()).unwrap();
    assert_eq!(
        loaded.det.as_ref().unwrap().model_path,
        dir.path().join("models/PP-OCRv6_det_small.onnx")
    );
    assert_eq!(
        loaded.cls.as_ref().unwrap().model_path,
        dir.path()
            .join("models/ch_ppocr_mobile_v2.0_cls_mobile.onnx")
    );
    assert_eq!(loaded.rec.as_ref().unwrap().model_path, absolute_model);
    assert_eq!(
        loaded.rec.as_ref().unwrap().dict_path,
        dir.path().join("../shared/dict.txt")
    );
    let before = loaded.to_toml_string().unwrap();
    loaded
        .resolve_model_paths(dir.path().join("different-base"))
        .unwrap();
    assert_eq!(loaded.to_toml_string().unwrap(), before);
}

#[test]
fn explicit_base_and_working_directory_are_independent_of_config_location() {
    let dir = TestDir::new("config-base");
    let path = dir.path().join("ocr.toml");
    fs::write(&path, config_text()).unwrap();
    let base = dir.path().join("deployment");
    let options =
        ConfigLoadOptions::new().with_model_path_base(ModelPathBase::Directory(base.clone()));
    let from_file = RapidOcrConfig::from_toml_file_with_options(&path, &options).unwrap();
    let from_str = RapidOcrConfig::from_toml_str_with_base(&config_text(), &base).unwrap();
    assert_eq!(
        from_file.to_toml_string().unwrap(),
        from_str.to_toml_string().unwrap()
    );
    assert_eq!(
        from_file.rec.unwrap().dict_path,
        base.join("models/ppocrv6_dict.txt")
    );

    let options = ConfigLoadOptions::new().with_model_path_base(ModelPathBase::WorkingDirectory);
    let config = RapidOcrConfig::from_toml_file_with_options(&path, &options).unwrap();
    assert_eq!(
        config.det.unwrap().model_path,
        std::env::current_dir()
            .unwrap()
            .join("models/PP-OCRv6_det_small.onnx")
    );
    let relative_base =
        RapidOcrConfig::from_toml_str_with_base(&config_text(), "deployment").unwrap();
    assert_eq!(
        relative_base.rec.unwrap().dict_path,
        std::env::current_dir()
            .unwrap()
            .join("deployment/models/ppocrv6_dict.txt")
    );
}

#[test]
fn compatible_file_options_preserve_unknown_fields_overrides_and_unresolved_paths() {
    let dir = TestDir::new("config-compatible");
    let path = dir.path().join("ocr.toml");
    let text = format!(
        "extension = true\nuse_det = false\nuse_cls = false\n{}",
        config_text()
    );
    fs::write(&path, text).unwrap();
    let old = RapidOcrConfig::from_toml_file(&path).unwrap();
    let explicit =
        RapidOcrConfig::from_toml_file_with_options(&path, &ConfigLoadOptions::compatible())
            .unwrap();
    assert_eq!(
        old.to_toml_string().unwrap(),
        explicit.to_toml_string().unwrap()
    );
    assert_eq!(explicit.pipeline, PipelineConfig::recognition_only());
    assert_eq!(
        explicit.rec.unwrap().dict_path,
        PathBuf::from("models/ppocrv6_dict.txt")
    );

    let options = ConfigLoadOptions::new().with_parse_mode(ConfigParseMode::Compatible);
    let resolved = RapidOcrConfig::from_toml_file_with_options(path, &options).unwrap();
    assert_eq!(
        resolved.rec.unwrap().dict_path,
        dir.path().join("models/ppocrv6_dict.txt")
    );
}

#[test]
fn strict_file_entry_preserves_paths_and_older_defaults() {
    let dir = TestDir::new("config-strict");
    let path = dir.path().join("ocr.toml");
    let mut value: toml::Value = config_text().parse().unwrap();
    value
        .get_mut("det")
        .unwrap()
        .as_table_mut()
        .unwrap()
        .remove("input_limits");
    value
        .get_mut("rec")
        .unwrap()
        .as_table_mut()
        .unwrap()
        .remove("input_limits");
    value.as_table_mut().unwrap().remove("inference");
    fs::write(&path, toml::to_string(&value).unwrap()).unwrap();
    let config = RapidOcrConfig::from_toml_file_strict(path).unwrap();
    assert_eq!(config.inference, Default::default());
    assert_eq!(
        config.det.as_ref().unwrap().input_limits,
        Default::default()
    );
    assert_eq!(
        config.rec.as_ref().unwrap().input_limits,
        Default::default()
    );
    assert_eq!(
        config.rec.unwrap().dict_path,
        PathBuf::from("models/ppocrv6_dict.txt")
    );
}

#[test]
fn failed_path_resolution_is_atomic_and_reports_the_field() {
    let dir = TestDir::new("config-path-errors");
    let mut config = RapidOcrConfig::ppocr_v6_small("models");
    config.rec.as_mut().unwrap().dict_path = PathBuf::new();
    let before = config.det.as_ref().unwrap().model_path.clone();
    let error = config.resolve_model_paths(dir.path()).unwrap_err();
    assert!(
        matches!(error, OcrError::Config { kind: ConfigErrorKind::PathResolution, field: Some(field), .. } if field == "rec.dict_path")
    );
    assert_eq!(config.det.unwrap().model_path, before);
    assert!(matches!(
        RapidOcrConfig::from_toml_str_with_base(&config_text(), "").unwrap_err(),
        OcrError::Config {
            kind: ConfigErrorKind::PathResolution,
            ..
        }
    ));
}

#[cfg(windows)]
#[test]
fn windows_drive_relative_and_root_relative_paths_require_explicit_drives() {
    let dir = TestDir::new("config-windows-paths");
    for path in [r"C:models\rec.onnx", r"\models\rec.onnx"] {
        let mut config = RapidOcrConfig::ppocr_v6_small("models");
        config.rec.as_mut().unwrap().model_path = path.into();
        let error = config.resolve_model_paths(dir.path()).unwrap_err();
        assert!(
            matches!(error, OcrError::Config { kind: ConfigErrorKind::PathResolution, field: Some(field), .. } if field == "rec.model_path")
        );
    }
}
