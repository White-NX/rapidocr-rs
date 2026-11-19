use rapidocr_core::{
    config::{
        ClsConfig, DetConfig, DetInputLimits, DetOverflowBehavior, ExecutionProvider,
        InferenceOptions, LimitType, PipelineConfig, RapidOcrConfig, RecConfig, RecInputLimits,
    },
    model::{
        ClsModelSpec, DetModelSpec, ModelAssetKind, ModelAssetSpec, ModelDownloadMode,
        ModelSetSpec, RecModelSpec,
    },
    types::{
        OcrLine, OcrOutput, OcrRotation, OcrTextAlignment, OcrTextMapping, OcrTimings, OcrToken,
        Quad, TimedOcrOutput,
    },
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
        .with_input_limits(RecInputLimits::new(Some(4096), Some(2 * 1024 * 1024)))
        .with_return_tokens(true);
    let config = RapidOcrConfig::new(Some(det), Some(cls), Some(rec))
        .with_pipeline(PipelineConfig::full())
        .with_inference_options(inference);

    config.validate().unwrap();
    assert!(config.rec.as_ref().unwrap().return_tokens);
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
    assert!(line.text_alignment.is_none());
    let value = serde_json::to_value(&line).unwrap();
    assert!(value["detection_score"].is_null());
    assert!(value["rotation"].is_null());
    assert!(value.get("text_alignment").is_none());

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
fn token_ranges_and_selection_coordinates_round_trip_through_json() {
    let bbox = Quad::from_xyxy(10.0, 20.0, 30.0, 25.0);
    let token = OcrToken::new(0..8, 0.9, 2..4, 0.3, [0.0, 1.0], bbox.clone());
    let line = OcrLine::new(bbox, "🇨🇳", 0.9).with_text_alignment(OcrTextAlignment::new(
        320,
        160,
        80,
        vec![token],
    ));
    let value = serde_json::to_value(&line).unwrap();
    assert!(value["text_alignment"].get("mapping").is_none());
    assert_eq!(
        value["text_alignment"]["tokens"][0]["text_range"],
        serde_json::json!({"start": 0, "end": 8})
    );
    let decoded: OcrLine = serde_json::from_value(value).unwrap();
    let alignment = decoded.text_alignment.unwrap();
    assert!(alignment.mapping.is_none());
    assert_eq!(alignment.input_width, 320);
    assert_eq!(alignment.valid_width, 160);
    let token = &alignment.tokens[0];
    assert_eq!(&decoded.text[token.text_range.clone()], "🇨🇳");
    assert_eq!(token.timestep_range, 2..4);
    assert_eq!(token.x_center, 0.3);
    assert_eq!(token.x_range, [0.0, 1.0]);
    assert_eq!(token.bbox.points, line.bbox.points);

    let mapping = OcrTextMapping::new(
        [[20.0, 0.0, 10.0], [0.0, 5.0, 20.0], [0.0, 0.0, 1.0]],
        [40, 30],
    );
    let alignment = alignment.with_mapping(mapping);
    let value = serde_json::to_value(&alignment).unwrap();
    assert_eq!(value["mapping"]["source_size"], serde_json::json!([40, 30]));
    assert_eq!(
        value["mapping"]["matrix"],
        serde_json::json!([[20.0, 0.0, 10.0], [0.0, 5.0, 20.0], [0.0, 0.0, 1.0]])
    );
    let decoded: OcrTextAlignment = serde_json::from_value(value).unwrap();
    let mapping = decoded.mapping.unwrap();
    assert_eq!(mapping.matrix, alignment.mapping.unwrap().matrix);
    assert_eq!(
        mapping
            .selection_quad(decoded.tokens[0].x_range)
            .unwrap()
            .points,
        line.bbox.points
    );
}

#[test]
fn source_mapping_projects_homogeneous_coordinates_and_clips_selection_cells() {
    let mapping = OcrTextMapping::new(
        [[30.0, 0.0, -10.0], [0.0, 20.0, -5.0], [0.5, 0.25, 1.0]],
        [11, 9],
    );
    let midpoint = mapping.project_point([0.5, 0.5]).unwrap();
    assert!((midpoint[0] - 40.0 / 11.0).abs() < 1e-5);
    assert!((midpoint[1] - 40.0 / 11.0).abs() < 1e-5);
    assert_eq!(mapping.project_point([0.0, 0.0]), Some([-10.0, -5.0]));
    assert_eq!(
        mapping.selection_quad([0.0, 1.0]).unwrap().points,
        [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]]
    );
    for range in [[0.5, 0.4], [-0.1, 0.2], [0.8, 1.1], [f32::NAN, 0.5]] {
        assert!(mapping.selection_quad(range).is_none());
    }
    assert!(mapping.project_point([f32::INFINITY, 0.0]).is_none());
    assert!(mapping.project_point([0.0, f32::NAN]).is_none());

    let zero_divisor =
        OcrTextMapping::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [11, 9]);
    assert!(zero_divisor.project_point([0.5, 0.5]).is_none());
    assert!(zero_divisor.selection_quad([0.0, 1.0]).is_none());

    let empty_image = OcrTextMapping::new(mapping.matrix, [0, 9]);
    assert!(empty_image.selection_quad([0.0, 1.0]).is_none());
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
