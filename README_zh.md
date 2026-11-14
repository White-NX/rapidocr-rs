<div style="text-align: center;">
    RapidOCR-rs
</div>

---

RapidOCR-rs 是一款基于[RapidOCR](https://github.com/RapidAI/RapidOCR)进行 Rust 重写的开源 OCR 解决方案。

需要 Rust 1.92 或更新版本。`ort` 与 `ort-sys` 固定为 `2.0.0-rc.15`，支持 Rust 1.100 的构建目录布局。

基于原项目开源的 PaddleOCR ONNX 模型，我们将其在不引入 `cv2` 的前提下移植到了 Rust，以方便开发者将 RapidOCR 轻松地集成到产品中，同时获得与 python 版本的 RapidOCR 相似的性能。

## 快速开始

### 在您的 Rust 项目中使用：

```bash
cargo add rapidocr-core
```

```rust
use rapidocr_core::{
    config::{InferenceOptions, PipelineConfig},
    model::{model_set_by_name, ModelCache, ModelDownloadMode},
    RapidOcr,
};

fn main() -> anyhow::Result<()> {
    let model_set = model_set_by_name("ppocrv6-small").unwrap();
    let cache = ModelCache::new("models");
    cache.ensure_model_set_for_pipeline(
        model_set,
        PipelineConfig::without_cls(),
        ModelDownloadMode::Missing,
    )?;

    let cfg = cache
        .config_for(model_set)
        .with_pipeline(PipelineConfig::without_cls())
        .with_inference_options(
            InferenceOptions::new()
                .with_intra_threads(1)
                .with_inter_threads(1)
                .with_parallel_execution(false),
        );
    let mut ocr = RapidOcr::from_config(cfg)?;
    let output = ocr.run_path("path/to/image.png")?;

    for line in output.lines {
        println!("{:.5}\t{}", line.score, line.text);
    }

    Ok(())
}
```

### 或是使用 cli 快速验证：

```bash
cargo run -p rapidocr-cli -- --image path\to\image.png --model-dir models
```

The runnable example is in [crates/rapidocr-core/examples/library_usage.rs](crates/rapidocr-core/examples/library_usage.rs)。

## 由应用部署运行时

模型下载与 ONNX Runtime 二进制下载分别由 `model-download` 和 `download-binaries`
控制，默认均启用。应用自行提供模型和运行时 DLL 时，可以使用：

```toml
rapidocr-core = { version = "0.3.0", default-features = false, features = ["load-dynamic", "tokio", "directml"] }
```

先调用 `OcrRuntime::load_from("runtime/onnxruntime.dll")`，再通过
`RapidOcr::new_with_runtime(config, &runtime)` 或
`TokioRapidOcr::new_with_runtime(config, runtime).await` 创建实例。
普通 `new` 构造函数自动共享同一个进程级运行时；调用 `shutdown().await` 会释放工作线程和模型会话。
已经由应用创建的 `ort` 环境可以通过 `OcrRuntime::from_environment` 接入。

本库明确选择 ONNX Runtime API 1.24，因此可继续使用应用提供的 1.24.x DLL。
默认二进制下载目前提供 1.31 运行时。应用直接依赖 `ort` 时，也应关闭其默认功能并选择
`api-24`，避免 Cargo 合并功能后提高运行时版本要求。运行时 DLL 应在其他 `ort` API 使用前选定。

## 模型校验与识别输入限制

模型加载时会校验输入与输出形状、字典和分类标签的数量，推理时会校验批次大小。
不匹配的模型返回错误。识别阶段默认限制单个批次的填充宽度为 4096 像素、
填充像素总数为 2097152，超过限制会在分配张量前返回错误。

```toml
[rec.input_limits]
max_width = 4096
max_batch_pixels = 2097152
```

应用可以切分长文本、减小批次、提高限制，或者使用 `RecInputLimits::unrestricted()`。
TOML 中可以用 `"unlimited"` 关闭某个检测或识别限制，省略限制表则使用默认值。
这些限制约束输入张量，不代表整个进程的内存上限。`ModelCache` 的下载接口会阻塞，
需要下载时应在进入 Tokio 前调用，或使用 `spawn_blocking`。
