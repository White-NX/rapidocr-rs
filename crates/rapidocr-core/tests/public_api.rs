use rapidocr_core::{
    config::{
        ClsConfig, DetConfig, DetInputLimits, DetOverflowBehavior, ExecutionProvider,
        InferenceOptions, LimitType, PipelineConfig, RapidOcrConfig, RecConfig, RecInputLimits,
    },
    model::{
        ClsModelSpec, DetModelSpec, ModelAssetKind, ModelAssetSpec, ModelDownloadMode,
        ModelSetSpec, RecModelSpec,
    },
    types::{OcrLine, OcrOutput, OcrRotation, OcrTimings, Quad, TimedOcrOutput},
};

const DET_ASSET: ModelAssetSpec = ModelAssetSpec::new(
    "test detector",
    ModelAssetKind::Detection,
    "det.onnx",
    "https://example.invalid/det.onnx",
    None,
);
const CLS_ASSET: ModelAssetSpec = ModelAssetSpec::new(
    "test classifier",
    ModelAssetKind::Classification,
    "cls.onnx",
    "https://example.invalid/cls.onnx",
    None,
);
const REC_ASSET: ModelAssetSpec = ModelAssetSpec::new(
    "test recognizer",
    ModelAssetKind::Recognition,
    "rec.onnx",
    "https://example.invalid/rec.onnx",
    None,
);
const DICT_ASSET: ModelAssetSpec = ModelAssetSpec::new(
    "test dictionary",
    ModelAssetKind::Dictionary,
    "dict.txt",
    "https://example.invalid/dict.txt",
    None,
);

const MODEL_SET: ModelSetSpec = ModelSetSpec::new(
    "test-model-set",
    "test-family",
    DetModelSpec::new(
        DET_ASSET,
        736,
        LimitType::Min,
        [0.5; 3],
        [0.5; 3],
        0.3,
        0.5,
        1000,
        1.6,
        3,
    ),
    ClsModelSpec::new(CLS_ASSET, [3, 48, 192], 6, 0.9, &["0", "180"]),
    RecModelSpec::new(REC_ASSET, DICT_ASSET, [3, 48, 320], 6),
);

#[test]
fn public_data_types_can_be_built_without_struct_literals() {
    let inference = InferenceOptions::new()
        .with_intra_threads(2)
        .with_inter_threads(1)
        .with_parallel_execution(false)
        .with_cpu_mem_arena(false)
        .with_execution_provider(ExecutionProvider::Cpu);
    let limits = DetInputLimits::new(
        Some(4096),
        Some(2048 * 2048),
        DetOverflowBehavior::Downscale,
    );
    let det = DetConfig::new(
        "models/det.onnx",
        736,
        LimitType::Min,
        [0.5; 3],
        [0.5; 3],
        0.3,
        0.5,
        1000,
        1.6,
        3,
    )
    .with_input_limits(limits);
    let cls = ClsConfig::new(
        "models/cls.onnx",
        [3, 48, 192],
        6,
        0.9,
        vec!["0".to_string(), "180".to_string()],
    );
    let rec = RecConfig::new("models/rec.onnx", "models/dict.txt", [3, 48, 320], 6)
        .with_input_limits(RecInputLimits::new(Some(4096), Some(2 * 1024 * 1024)));
    let config = RapidOcrConfig::new(Some(det), Some(cls), Some(rec))
        .with_pipeline(PipelineConfig::full())
        .with_inference_options(inference);

    config.validate().unwrap();
    assert_eq!(MODEL_SET.name, "test-model-set");

    let bbox = Quad::new([[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]]);
    let line = OcrLine::new(bbox, "text", 0.95)
        .with_detection_score(0.85)
        .with_rotation(OcrRotation::new("180", 0.99, 180));
    let output = OcrOutput::new(vec![line]);
    let timed = TimedOcrOutput::new(output, OcrTimings::default());

    assert_eq!(timed.output.lines[0].text, "text");
}

#[test]
fn result_metadata_round_trips_and_legacy_json_remains_readable() {
    let legacy = serde_json::json!({
        "bbox": {"points": [[0, 0], [10, 0], [10, 4], [0, 4]]},
        "text": "text",
        "score": 0.95
    });
    let line: OcrLine = serde_json::from_value(legacy).unwrap();
    assert!(line.detection_score.is_none());
    assert!(line.rotation.is_none());
    let value = serde_json::to_value(&line).unwrap();
    assert!(value["detection_score"].is_null());
    assert!(value["rotation"].is_null());

    let line = line
        .with_detection_score(0.85)
        .with_rotation(OcrRotation::new("180", 0.9, 0));
    let value = serde_json::to_value(&line).unwrap();
    assert_eq!(value["rotation"]["label"], "180");
    assert_eq!(value["rotation"]["applied_degrees"], 0);
    let decoded: OcrLine = serde_json::from_value(value).unwrap();
    assert_eq!(decoded.score, line.score);
    assert_eq!(decoded.detection_score, line.detection_score);
    let rotation = decoded.rotation.unwrap();
    assert_eq!(rotation.label, "180");
    assert_eq!(rotation.score, 0.9);
    assert_eq!(rotation.applied_degrees, 0);
}

#[test]
fn extensible_enums_are_matched_with_fallbacks() {
    assert_eq!(execution_provider_name(ExecutionProvider::Cpu), "cpu");
    assert_eq!(limit_type_name(LimitType::Max), "max");
    assert_eq!(
        overflow_behavior_name(DetOverflowBehavior::Reject),
        "reject"
    );
    assert_eq!(asset_kind_name(ModelAssetKind::Dictionary), "dictionary");
    assert_eq!(download_mode_name(ModelDownloadMode::Never), "never");
}

fn execution_provider_name(value: ExecutionProvider) -> &'static str {
    match value {
        ExecutionProvider::Cpu => "cpu",
        ExecutionProvider::DirectMl => "direct-ml",
        _ => "unknown",
    }
}

fn limit_type_name(value: LimitType) -> &'static str {
    match value {
        LimitType::Min => "min",
        LimitType::Max => "max",
        _ => "unknown",
    }
}

fn overflow_behavior_name(value: DetOverflowBehavior) -> &'static str {
    match value {
        DetOverflowBehavior::Downscale => "downscale",
        DetOverflowBehavior::Reject => "reject",
        DetOverflowBehavior::Allow => "allow",
        _ => "unknown",
    }
}

fn asset_kind_name(value: ModelAssetKind) -> &'static str {
    match value {
        ModelAssetKind::Detection => "detection",
        ModelAssetKind::Classification => "classification",
        ModelAssetKind::Recognition => "recognition",
        ModelAssetKind::Dictionary => "dictionary",
        _ => "unknown",
    }
}

fn download_mode_name(value: ModelDownloadMode) -> &'static str {
    match value {
        ModelDownloadMode::Missing => "missing",
        ModelDownloadMode::Never => "never",
        _ => "unknown",
    }
}
