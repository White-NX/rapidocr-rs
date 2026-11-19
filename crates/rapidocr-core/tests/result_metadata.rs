mod common;

use image::RgbImage;
use rapidocr_core::{
    config::{ClsConfig, DetConfig, LimitType, PipelineConfig, RapidOcrConfig, RecConfig},
    types::OcrOutput,
    RapidOcr,
};

fn detection_config() -> RapidOcrConfig {
    RapidOcrConfig::new(
        Some(DetConfig::new(
            common::model_path("scored-det.onnx"),
            128,
            LimitType::Max,
            [0.5; 3],
            [0.5; 3],
            0.3,
            0.3,
            10,
            1.6,
            3,
        )),
        None,
        None,
    )
}

fn paired_config(classification: bool) -> RapidOcrConfig {
    let mut config = detection_config();
    config.rec = Some(RecConfig::new(
        common::model_path("paired-rec.onnx"),
        common::model_path("one-token.txt"),
        [3, 48, 320],
        2,
    ));
    if classification {
        config.cls = Some(ClsConfig::new(
            common::model_path("paired-cls.onnx"),
            [3, 48, 192],
            2,
            0.8,
            vec!["0".into(), "180".into()],
        ));
    }
    config.pipeline = PipelineConfig::new(true, classification, true);
    config
}

fn run(config: RapidOcrConfig) -> OcrOutput {
    RapidOcr::new(config)
        .unwrap()
        .run_image(&RgbImage::new(128, 128))
        .unwrap()
}

#[test]
fn detection_only_exposes_confidence_without_repurposing_recognition_score() {
    let output = run(detection_config());
    assert_eq!(output.lines.len(), 2);
    for line in &output.lines {
        assert!(line.text.is_empty());
        assert_eq!(line.score, 0.0);
        let score = line.detection_score.unwrap();
        assert!(score > 0.3 && score <= 1.0);
        assert!(line.rotation.is_none());
    }
    assert!(output.lines[0].bbox.points[0][1] < output.lines[1].bbox.points[0][1]);
    assert!(output.lines[0].detection_score > output.lines[1].detection_score);
}

#[test]
fn metadata_stays_with_its_box_after_batch_reordering_and_output_filtering() {
    let detected = run(detection_config());
    let mut config = paired_config(true);
    config.text_score = 0.0;
    let unfiltered = run(config);
    assert_eq!(unfiltered.lines.len(), 2);
    for (line, detection) in unfiltered.lines.iter().zip(&detected.lines) {
        assert_eq!(line.bbox.points, detection.bbox.points);
        assert_eq!(line.detection_score, detection.detection_score);
        assert_eq!(line.text, "A");
    }
    assert_eq!(unfiltered.lines[0].score, 0.2);
    assert_eq!(unfiltered.lines[1].score, 0.96);
    let upright = unfiltered.lines[0].rotation.as_ref().unwrap();
    assert_eq!(upright.label, "0");
    assert_eq!(upright.score, 0.92);
    assert_eq!(upright.applied_degrees, 0);
    let rotated = unfiltered.lines[1].rotation.as_ref().unwrap();
    assert_eq!(rotated.label, "180");
    assert_eq!(rotated.score, 0.84);
    assert_eq!(rotated.applied_degrees, 180);

    let filtered = run(paired_config(true));
    assert_eq!(filtered.lines.len(), 1);
    let line = &filtered.lines[0];
    assert_eq!(line.bbox.points, detected.lines[1].bbox.points);
    assert_eq!(line.detection_score, detected.lines[1].detection_score);
    assert_eq!(line.score, 0.96);
    let rotation = line.rotation.as_ref().unwrap();
    assert_eq!(rotation.label, rotated.label);
    assert_eq!(rotation.score, rotated.score);
    assert_eq!(rotation.applied_degrees, rotated.applied_degrees);
}

#[test]
fn disabled_stages_have_no_metadata() {
    let recognized = run(common::recognition_config("constant-rec.onnx"));
    assert_eq!(recognized.lines[0].text, "A");
    assert!(recognized.lines[0].detection_score.is_none());
    assert!(recognized.lines[0].rotation.is_none());

    let detected_and_recognized = run(paired_config(false));
    assert_eq!(detected_and_recognized.lines.len(), 1);
    assert!(detected_and_recognized.lines[0].detection_score.is_some());
    assert!(detected_and_recognized.lines[0].rotation.is_none());
}

#[test]
fn rec_only_keeps_classification_below_or_at_the_rotation_threshold() {
    for threshold in [0.8, 0.9, 1.0] {
        let mut config = common::recognition_config("constant-rec.onnx");
        config.cls = Some(ClsConfig::new(
            common::model_path("constant-cls.onnx"),
            [3, 48, 192],
            1,
            threshold,
            vec!["0".into(), "180".into()],
        ));
        config.pipeline.use_cls = true;
        let output = run(config);
        let line = &output.lines[0];
        assert!(line.detection_score.is_none());
        assert_eq!(line.score, 1.0);
        let rotation = line.rotation.as_ref().unwrap();
        assert_eq!(rotation.label, "180");
        assert_eq!(rotation.score, 0.9);
        assert_eq!(
            rotation.applied_degrees,
            if threshold < 0.9 { 180 } else { 0 }
        );
    }
}

#[test]
fn empty_detection_output_still_succeeds_with_classification_enabled() {
    let mut config = paired_config(true);
    config.det.as_mut().unwrap().model_path = common::model_path("constant-det.onnx");
    assert!(run(config).lines.is_empty());
}

#[test]
fn opt_in_token_alignment_preserves_default_recognition_json() {
    let image = RgbImage::new(197, 59);
    let mut config = common::recognition_config("constant-rec.onnx");
    let plain = RapidOcr::new(config.clone())
        .unwrap()
        .run_image(&image)
        .unwrap();
    assert!(plain.lines[0].text_alignment.is_none());
    let plain_json = serde_json::to_value(&plain).unwrap();
    assert!(plain_json["lines"][0].get("text_alignment").is_none());

    config.rec.as_mut().unwrap().return_tokens = true;
    let output = RapidOcr::new(config).unwrap().run_image(&image).unwrap();
    let line = &output.lines[0];
    let alignment = line.text_alignment.as_ref().unwrap();
    assert_eq!(alignment.input_width, 320);
    assert_eq!(alignment.valid_width, 161);
    assert_eq!(alignment.time_steps, 2);
    assert_eq!(alignment.tokens.len(), 1);
    let mapping = alignment.mapping.as_ref().unwrap();
    assert_eq!(mapping.source_size, [197, 59]);
    assert_eq!(
        mapping.matrix,
        [[196.0, 0.0, 0.0], [0.0, 58.0, 0.0], [0.0, 0.0, 1.0]]
    );
    let token = &alignment.tokens[0];
    assert_eq!(&line.text[token.text_range.clone()], "A");
    assert_eq!(token.text_range, 0..1);
    assert_eq!(token.timestep_range, 0..1);
    assert_eq!(token.score, line.score);
    assert_eq!(token.x_range, [0.0, 1.0]);
    assert_eq!(token.bbox.points, line.bbox.points);
    assert_eq!(
        mapping.selection_quad(token.x_range).unwrap().points,
        token.bbox.points
    );
    let mut output_json = serde_json::to_value(&output).unwrap();
    output_json["lines"][0]
        .as_object_mut()
        .unwrap()
        .remove("text_alignment");
    assert_eq!(output_json, plain_json);
}

#[test]
fn token_cells_keep_their_line_and_orientation_after_filtering() {
    let mut config = paired_config(true);
    config.text_score = 0.0;
    let plain = run(config.clone());
    config.rec.as_mut().unwrap().return_tokens = true;
    let output = run(config.clone());
    for (line, plain) in output.lines.iter().zip(&plain.lines) {
        assert_eq!(line.bbox.points, plain.bbox.points);
        assert_eq!(line.text, plain.text);
        assert_eq!(line.score, plain.score);
        let alignment = line.text_alignment.as_ref().unwrap();
        let token = &alignment.tokens[0];
        let mapping = alignment.mapping.as_ref().unwrap();
        assert_eq!(mapping.source_size, [128, 128]);
        let reconstructed = mapping.selection_quad(token.x_range).unwrap();
        for (actual, expected) in reconstructed.points.iter().zip(token.bbox.points) {
            assert!((actual[0] - expected[0]).abs() < 1e-3);
            assert!((actual[1] - expected[1]).abs() < 1e-3);
        }
        assert_eq!(token.score, line.score);
        assert_eq!(&line.text[token.text_range.clone()], "A");
        assert!(token
            .bbox
            .points
            .iter()
            .all(|[x, y]| (0.0..=127.0).contains(x) && (0.0..=127.0).contains(y)));
    }
    let upright = &output.lines[0].text_alignment.as_ref().unwrap().tokens[0]
        .bbox
        .points;
    assert!(upright[0][0] < upright[1][0]);
    let rotated = &output.lines[1].text_alignment.as_ref().unwrap().tokens[0]
        .bbox
        .points;
    assert!(rotated[0][0] > rotated[1][0]);
    assert!(rotated[0][1] > rotated[3][1]);

    config.text_score = 0.5;
    let filtered = run(config.clone());
    assert_eq!(filtered.lines.len(), 1);
    assert_eq!(
        serde_json::to_value(&filtered.lines[0]).unwrap(),
        serde_json::to_value(&output.lines[1]).unwrap()
    );

    config.pipeline = PipelineConfig::detection_only();
    assert!(run(config)
        .lines
        .iter()
        .all(|line| line.text_alignment.is_none()));
}
