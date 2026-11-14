mod common;

use image::RgbImage;
use rapidocr_core::{OcrRuntime, RapidOcr};

// Kept in a separate integration-test binary: ORT environments are process-wide.
#[test]
fn application_environment_is_shared_without_initializing_another_environment() {
    #[cfg(feature = "load-dynamic")]
    let builder = ort::init_from(std::env::var_os("ORT_DYLIB_PATH").unwrap_or_else(|| {
        format!(
            "{}onnxruntime{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        )
        .into()
    }))
    .unwrap();
    #[cfg(not(feature = "load-dynamic"))]
    let builder = ort::init();

    let environment = builder
        .with_name("host-application")
        .with_telemetry(false)
        .build()
        .unwrap();
    let runtime = OcrRuntime::from_environment(environment);
    let config = common::recognition_config("constant-rec.onnx");
    let mut ocr = RapidOcr::new_with_runtime(config.clone(), &runtime).unwrap();
    assert_eq!(
        ocr.run_image(&RgbImage::new(320, 48)).unwrap().lines[0].text,
        "A"
    );

    #[cfg(feature = "tokio")]
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(async move {
            let service = rapidocr_core::TokioRapidOcr::new_with_runtime(config, runtime)
                .await
                .unwrap();
            let output = service
                .run_image(std::sync::Arc::new(RgbImage::new(320, 48)))
                .await
                .unwrap();
            assert_eq!(output.output.lines[0].text, "A");
            service.shutdown().await.unwrap();
        });
}
