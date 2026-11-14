mod common;

use image::RgbImage;
use ort::AsPointer;
use rapidocr_core::{config::RecInputLimits, OcrRuntime, RapidOcr};

#[test]
fn detection_and_recognition_sessions_share_the_default_environment() {
    use rapidocr_core::config::{DetConfig, LimitType, RapidOcrConfig};

    let rec = common::recognition_config("constant-rec.onnx").rec;
    let config = RapidOcrConfig::new(
        Some(DetConfig::new(
            common::model_path("constant-det.onnx"),
            32,
            LimitType::Max,
            [0.5; 3],
            [0.5; 3],
            0.3,
            0.5,
            10,
            1.6,
            3,
        )),
        None,
        rec,
    );
    let mut ocr = RapidOcr::new(config).unwrap();
    assert!(ocr
        .run_image(&RgbImage::new(320, 48))
        .unwrap()
        .lines
        .is_empty());
}

#[test]
fn default_runtime_survives_concurrent_initialization_and_pipeline_recreation() {
    let runtime = OcrRuntime::new().unwrap();
    let address = runtime.environment().ptr() as usize;
    drop(runtime);
    let workers = (0..4)
        .map(|_| std::thread::spawn(|| OcrRuntime::new().unwrap().environment().ptr() as usize))
        .collect::<Vec<_>>();
    for worker in workers {
        assert_eq!(worker.join().unwrap(), address);
    }

    for _ in 0..3 {
        let mut ocr = RapidOcr::new(common::recognition_config("constant-rec.onnx")).unwrap();
        let output = ocr.run_image(&RgbImage::new(320, 48)).unwrap();
        assert_eq!(output.lines[0].text, "A");
    }
}

#[test]
fn incompatible_dictionary_is_rejected_when_loading_the_model() {
    let mut config = common::recognition_config("constant-rec.onnx");
    config.rec.as_mut().unwrap().dict_path = common::model_path("two-tokens.txt");
    let error = match RapidOcr::new(config) {
        Ok(_) => panic!("incompatible dictionary was accepted"),
        Err(error) => error,
    };
    assert!(format!("{error:#}").contains("output dimension 2 must be 4, got 3"));
}

#[test]
fn invalid_model_batch_returns_an_error_instead_of_indexing_past_crops() {
    let mut ocr = RapidOcr::new(common::recognition_config("wrong-batch-rec.onnx")).unwrap();
    let error = ocr.run_image(&RgbImage::new(320, 48)).unwrap_err();
    assert!(format!("{error:#}").contains("output batch size must match input batch size 1, got 2"));
}

#[test]
fn long_line_limits_are_explicit_and_can_be_disabled() {
    let mut config = common::recognition_config("constant-rec.onnx");
    let image = RgbImage::new(200, 1);
    let mut limited = RapidOcr::new(config.clone()).unwrap();
    let error = limited.run_image(&image).unwrap_err();
    assert!(format!("{error:#}").contains("rec.input_limits.max_width=4096"));

    config.rec.as_mut().unwrap().input_limits = RecInputLimits::unrestricted();
    let mut unrestricted = RapidOcr::new(config).unwrap();
    assert_eq!(unrestricted.run_image(&image).unwrap().lines[0].text, "A");
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn independent_tokio_services_share_runtime_and_release_workers() {
    use rapidocr_core::TokioRapidOcr;
    use std::sync::Arc;

    let config = common::recognition_config("constant-rec.onnx");
    let first = TokioRapidOcr::new(config.clone()).await.unwrap();
    let second = TokioRapidOcr::new(config).await.unwrap();
    let image = Arc::new(RgbImage::new(320, 48));
    assert_eq!(
        first.run_image(image.clone()).await.unwrap().output.lines[0].text,
        "A"
    );
    assert_eq!(
        second.run_image(image).await.unwrap().output.lines[0].text,
        "A"
    );
    first.shutdown().await.unwrap();
    second.shutdown().await.unwrap();
}
