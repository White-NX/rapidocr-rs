#![cfg(feature = "load-dynamic")]

mod common;

use ort::AsPointer;
use rapidocr_core::OcrRuntime;

#[test]
fn failed_library_load_can_be_retried_and_runtime_selection_is_stable() {
    let invalid_library = common::model_path("constant-rec.onnx");
    let error = OcrRuntime::load_from(&invalid_library).unwrap_err();
    assert!(format!("{error:#}").contains("failed to load ONNX Runtime"));

    let library = std::env::var_os("ORT_DYLIB_PATH")
        .expect("dynamic runtime tests require ORT_DYLIB_PATH to point to a runtime library");
    let first = OcrRuntime::load_from(&library).unwrap();
    let second = OcrRuntime::load_from(&library).unwrap();
    assert_eq!(first.environment().ptr(), second.environment().ptr());
    assert_eq!(
        first.environment().ptr(),
        OcrRuntime::new().unwrap().environment().ptr()
    );

    let error = OcrRuntime::load_from(invalid_library).unwrap_err();
    assert!(format!("{error:#}").contains("already initialized with another library"));

    let mut ocr = rapidocr_core::RapidOcr::new_with_runtime(
        common::recognition_config("constant-rec.onnx"),
        &first,
    )
    .unwrap();
    assert_eq!(
        ocr.run_image(&image::RgbImage::new(320, 48)).unwrap().lines[0].text,
        "A"
    );
}
