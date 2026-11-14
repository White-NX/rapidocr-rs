mod common;
#[path = "common/temp_dir.rs"]
mod temp_dir;

use std::{error::Error, fs};

use image::RgbImage;
use rapidocr_core::{
    InputLimitKind, ModelErrorKind, OcrCancellationToken, OcrError, OcrErrorKind, OcrResult,
    OcrStage, RapidOcr, RuntimeErrorKind,
};
use temp_dir::TestDir;

#[test]
fn missing_model_and_dictionary_are_typed_and_keep_io_causes() {
    let dir = TestDir::new("missing-assets");
    for dictionary in [false, true] {
        let mut config = common::recognition_config("constant-rec.onnx");
        let path = dir.path().join(if dictionary {
            "missing.txt"
        } else {
            "missing.onnx"
        });
        if dictionary {
            config.rec.as_mut().unwrap().dict_path = path.clone();
        } else {
            config.rec.as_mut().unwrap().model_path = path.clone();
        }
        let error = RapidOcr::new(config).err().unwrap();
        assert_eq!(error.kind(), OcrErrorKind::Model);
        assert!(
            matches!(&error, OcrError::Model { kind: ModelErrorKind::NotFound, path: Some(asset), .. } if asset == &path)
        );
        assert_eq!(
            error
                .source()
                .unwrap()
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .kind(),
            std::io::ErrorKind::NotFound
        );
    }
}

#[test]
fn corrupt_model_preserves_the_onnx_error_and_its_code() {
    let dir = TestDir::new("corrupt-model");
    let path = dir.path().join("corrupt.onnx");
    fs::write(&path, b"not an ONNX protobuf").unwrap();
    let mut config = common::recognition_config("constant-rec.onnx");
    config.rec.as_mut().unwrap().model_path = path.clone();
    let error = RapidOcr::new(config).err().unwrap();
    assert!(
        matches!(&error, OcrError::Model { kind: ModelErrorKind::Incompatible, path: Some(asset), .. } if asset == &path)
    );
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<ort::Error>()
            .unwrap()
            .code(),
        ort::ErrorCode::InvalidProtobuf
    );
}

#[test]
fn backend_execution_failure_has_a_runtime_category_and_keeps_its_cause() {
    let mut ocr = RapidOcr::new(common::recognition_config("failing-rec.onnx")).unwrap();
    let result: OcrResult<_> = ocr.run_image(&RgbImage::new(320, 48));
    let error = result.unwrap_err();
    assert!(matches!(
        &error,
        OcrError::Runtime {
            kind: RuntimeErrorKind::Execution,
            ..
        }
    ));
    assert!(error.source().unwrap().is::<ort::Error>());
}

#[test]
fn invalid_images_and_cancelled_requests_have_distinct_categories() {
    let mut ocr = RapidOcr::new(common::recognition_config("constant-rec.onnx")).unwrap();
    let error = ocr.run_image(&RgbImage::new(0, 48)).unwrap_err();
    assert_eq!(error.kind(), OcrErrorKind::InvalidInput);
    let dir = TestDir::new("invalid-image");
    let path = dir.path().join("corrupt.png");
    fs::write(&path, b"not an image").unwrap();
    let error = ocr.run_path(&path).unwrap_err();
    assert!(matches!(&error, OcrError::InvalidInput { path: Some(input), .. } if input == &path));
    assert!(error.source().unwrap().is::<image::ImageError>());

    let cancellation = OcrCancellationToken::new();
    cancellation.cancel();
    let error = ocr
        .run_image_cancellable(&RgbImage::new(0, 0), &cancellation)
        .unwrap_err();
    assert!(matches!(error, OcrError::Cancelled(_)));
    assert!(rapidocr_core::is_cancelled_error(&error));
}

#[test]
fn oversized_image_header_is_rejected_before_pixel_allocation() {
    let dir = TestDir::new("image-allocation-limit");
    let path = dir.path().join("oversized.bmp");
    let mut bytes = Vec::new();
    image::codecs::bmp::BmpEncoder::new(&mut bytes)
        .encode(&[255, 255, 255], 1, 1, image::ExtendedColorType::Rgb8)
        .unwrap();
    // A small file claims a 20000 x 20000 image. No large buffer is created.
    bytes[18..22].copy_from_slice(&20000_u32.to_le_bytes());
    bytes[22..26].copy_from_slice(&20000_u32.to_le_bytes());
    fs::write(&path, bytes).unwrap();
    let mut ocr = RapidOcr::new(common::recognition_config("constant-rec.onnx")).unwrap();
    let error = ocr.run_path(&path).unwrap_err();
    assert!(
        matches!(
            &error,
            OcrError::InputLimit {
                stage: OcrStage::Preprocessing,
                kind: InputLimitKind::ImageBytes,
                requested: Some(1200000000),
                limit: Some(536870912),
                ..
            }
        ),
        "unexpected error: {error:?}"
    );
    assert!(error.source().unwrap().is::<image::ImageError>());
}

#[test]
fn oversized_recognition_exposes_recovery_metadata_without_matching_text() {
    let mut ocr = RapidOcr::new(common::recognition_config("constant-rec.onnx")).unwrap();
    let error = ocr.run_image(&RgbImage::new(200, 1)).unwrap_err();
    assert!(matches!(
        error,
        OcrError::InputLimit {
            stage: OcrStage::Recognition,
            kind: InputLimitKind::Width,
            requested: Some(9600),
            limit: Some(4096),
            ..
        }
    ));
}

#[test]
fn cached_checksum_failure_is_a_model_integrity_error_and_preserves_the_asset() {
    use rapidocr_core::{
        config::PipelineConfig,
        model::{ModelCache, ModelDownloadMode, PPOCRV6_SMALL},
    };
    let dir = TestDir::new("cache-integrity");
    let cache = ModelCache::new(dir.path());
    let path = cache.asset_path(PPOCRV6_SMALL.det.asset);
    fs::write(&path, b"corrupted model").unwrap();
    let error = cache
        .ensure_model_set_for_pipeline(
            &PPOCRV6_SMALL,
            PipelineConfig::detection_only(),
            ModelDownloadMode::Never,
        )
        .unwrap_err();
    assert!(
        matches!(error, OcrError::Model { kind: ModelErrorKind::Integrity, path: Some(asset), .. } if asset == path)
    );
    assert_eq!(fs::read(&path).unwrap(), b"corrupted model");
}

#[cfg(not(feature = "model-download"))]
#[test]
fn unavailable_downloader_has_a_model_category_without_attempting_network_access() {
    use rapidocr_core::{
        config::PipelineConfig,
        model::{ModelCache, ModelDownloadMode, PPOCRV6_SMALL},
    };
    let dir = TestDir::new("no-downloader");
    let cache = ModelCache::new(dir.path());
    let error = cache
        .ensure_model_set_for_pipeline(
            &PPOCRV6_SMALL,
            PipelineConfig::detection_only(),
            ModelDownloadMode::Missing,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        OcrError::Model {
            kind: ModelErrorKind::DownloadDisabled,
            ..
        }
    ));
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn tokio_preserves_the_same_input_limit_category() {
    use rapidocr_core::{TokioOcrError, TokioRapidOcr};
    use std::sync::Arc;

    let service = TokioRapidOcr::new(common::recognition_config("constant-rec.onnx"))
        .await
        .unwrap();
    let error = service
        .run_image(Arc::new(RgbImage::new(200, 1)))
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        TokioOcrError::Ocr(OcrError::InputLimit {
            stage: OcrStage::Recognition,
            kind: InputLimitKind::Width,
            ..
        })
    ));
    service.shutdown().await.unwrap();
}
