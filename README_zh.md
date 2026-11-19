<table>
<tr>
<td width="74%" valign="top">

# RapidOCR-rs

RapidOCR-rs 是一款基于[RapidOCR](https://github.com/RapidAI/RapidOCR)进行 Rust 重写的开源 OCR 解决方案。

需要 Rust 1.92 或更新版本。`ort` 与 `ort-sys` 固定为 `2.0.0-rc.15`，支持 Rust 1.100 的构建目录布局。

基于原项目开源的 PaddleOCR ONNX 模型，我们将其在不引入 `cv2` 的前提下移植到了 Rust，以方便开发者将 RapidOCR 轻松地集成到产品中，同时获得与 python 版本的 RapidOCR 相似的性能。

</td>
<td width="26%" valign="top">

<picture><img src="./docs/rapi-chan.png" width="100%" alt="" /></picture>

</td>
</tr>
</table>

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

fn main() -> rapidocr_core::OcrResult<()> {
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

## 结果分数与旋转信息

`types::OcrLine` 分别返回各阶段的结果：

| 字段 | 含义 |
| --- | --- |
| `score` | 平均识别置信度；关闭识别时为 `0.0`。 |
| `detection_score` | 检测框扩张前的 DB 分数；关闭检测时为 `None`。 |
| `rotation` | `OcrRotation` 包含分类标签 `label`、分类置信度 `score` 和实际旋转角度 `applied_degrees`；关闭分类时为 `None`。 |

仅检测模式返回空文本及有效的 `detection_score`。即使分类标签为 `"180"`，
置信度没有严格超过 `cls.thresh` 时，`applied_degrees` 仍为 `0`。
该角度只记录方向分类额外施加的裁剪图旋转（`0` 或 `180` 度），不包含
EXIF 方向校正或竖长检测框裁剪时的 90 度旋转。输出框仍使用原图坐标。
识别批次重排和结果过滤后，这些信息仍与各自的检测框对应。
JSON 中缺失阶段的结果为 `null`，也可继续读取不含新字段的旧 JSON。

CLI 保留识别分数、文本、检测框三列，并为已运行的阶段追加带名称的
`det_score`、`cls_label`、`cls_score` 与 `rotation`。

从 `0.2.2` 升级的指引集中在 [v0.3.0 发布说明](https://github.com/White-NX/rapidocr-rs/releases/tag/v0.3.0)。

## 可选的 token 对齐与文本选择

本功能自 `0.3.1` 提供，详见 [v0.3.1 发布说明](https://github.com/White-NX/rapidocr-rs/releases/tag/v0.3.1)。

通过 `RecConfig::with_return_tokens(true)`，或在现有 TOML `[rec]` 表中设置
`return_tokens = true`，可获得每行的 `text_alignment`。CLI 使用 `--return-tokens`，
在识别行末追加 `text_alignment=<JSON>`。默认关闭；未开启或仅检测时，该字段为
`None`，并从 JSON 中省略，原有文本、分数与行框输出保持一致。

`OcrTextAlignment` 返回批次填充宽度 `input_width`、该行实际缩放宽度
`valid_width`、CTC 输出时间步数 `time_steps` 和按解码顺序排列的 `tokens`。
每个 `OcrToken` 包含：

- `text_range`：行文本中的半开 **UTF-8 字节范围**，JSON 使用 `{ "start": ..., "end": ... }`。
- `score` 与 `timestep_range`：首个保留时间步的识别置信度，以及连续重复的非 blank 标签跨度。
- `x_center` 与 `x_range`：在识别图朝向下归一化到 `[0, 1]` 的对齐中心和估算选择区间。
- `bbox`：经实际裁切、旋转、缩放及补边的逆变换后，裁剪到原图边界的四边形。

空格和国旗 emoji 等多码点词表 token 会完整保留，各 token 的文本范围可以拼回
完整识别文本。Rust 可使用 `&line.text[token.text_range.clone()]` 获取文本；
JavaScript 若使用字符串索引，需要显式将 UTF-8 字节偏移转换为 UTF-16 偏移。

选择区间的内部边界取相邻 CTC 对齐中心的中点，首尾延伸到裁切图边缘。
归一化坐标使用实际缩放和批次填充宽度，并遵循竖长图 270° 旋转及分类器
180° 旋转后的识别朝向；原图框的角点顺序也保留这一朝向。

开启后，每行还会返回 `alignment.mapping`，包含按行存放的 `3x3` `f32` 矩阵
和原图 `[width, height]`。矩阵把旋转后的归一化裁切坐标映射为**未经边界裁剪**
的原图坐标；`selection_quad` 可以重建选择框，再裁剪到原图边界。
每行保存一个矩阵及各 token 的一维区间，即可避免保存每个 token 的四个角点。
存储时应保留完整的 `f32` 系数；重建与直接输出的角点可能有细小浮点舍入差异。
旧的对齐 JSON 即使不含 `mapping` 也能读取，原有构造函数保持可用。
[紧凑选择坐标说明](crates/rapidocr-core/README.md#compact-selection-geometry)
给出了矩阵乘法、角点顺序及裁剪约定。

这些框覆盖整行高度，属于**估算的选择单元格**；CTC 置信度不能说明定位精度。
标点宽度、大空隙与识别错误仍可能影响边界，补边中产生的 token 也可能被裁成零宽。
无需额外模型或 Python 运行时。[英文 API 说明](crates/rapidocr-core/README.md)
包含坐标公式及详细约定。

## 由应用部署运行时

模型下载与 ONNX Runtime 二进制下载分别由 `model-download` 和 `download-binaries`
控制，默认均启用。应用自行提供模型和运行时 DLL 时，可以使用：

```toml
rapidocr-core = { version = "0.3.1", default-features = false, features = ["load-dynamic", "tokio", "directml"] }
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

## 可分类的公共错误

同步 OCR、模型缓存、运行时和配置接口统一返回 `OcrResult<T>`，即
`Result<T, OcrError>`。调用方可以匹配错误变体或使用 `error.kind()`，无需匹配错误文本：

| 错误 | 处理依据 |
| --- | --- |
| `OcrError::Model` | `ModelErrorKind` 区分缺失、校验失败、不兼容、I/O 和下载错误；`path` 标识相关模型或字典。 |
| `OcrError::Runtime` | `RuntimeErrorKind` 区分初始化、会话配置、推理和取消信号错误；检查原始原因后决定是否重试。 |
| `OcrError::InputLimit` | `stage`、`kind`、`requested`、`limit` 标识超限资源，便于切图、减小批次或调整限制。 |
| `OcrError::InvalidInput` | 修复图片或路径后再运行。 |
| `OcrError::Config` | `ConfigErrorKind` 区分解析、未知字段、开关冲突、校验、I/O 和路径解析错误。 |
| `OcrError::Cancelled` | 停止当前请求；`is_cancelled_error` 接受 `&OcrError`。 |

底层 ONNX、图片解码、网络和 I/O 错误通过 `std::error::Error::source` 保留，
包括 ONNX 错误码。Tokio 使用 `TokioOcrError::Ocr(OcrError)` 返回同一套类型，
队列、取消、超时和工作线程状态仍有明确变体。返回类型的调整属于 `0.3.0` 的
API 兼容边界，核心库已移除对 `anyhow` 的依赖。

文件加载会在分配像素缓冲区前检查解码字节数，采用 `ImageReader` 默认的 512 MiB
分配限制，并以 `InputLimitKind::ImageBytes` 返回超限错误。需要自定义图片加载策略时，
可自行加载 RGB 图片并调用 `run_image`。

## 配置解析与路径基准

部署时建议显式选择解析和路径策略：

```rust
use rapidocr_core::config::{ConfigLoadOptions, RapidOcrConfig};

let config = RapidOcrConfig::from_toml_file_with_options(
    "settings/ocr.toml",
    &ConfigLoadOptions::new(),
)?;
```

默认使用严格解析，模型与识别字典的相对路径以配置文件目录为基准。
严格模式会拒绝各级未知字段，包括未启用阶段的字段。旧式顶层
`use_det`、`use_cls`、`use_rec` 仍可单独使用；同时写入 `[pipeline]` 时，
对应值必须一致。省略可选设置时仍使用原有默认值。

原有 `from_toml_str`、`from_toml_file` 继续容忍未知字段，让顶层开关覆盖
`[pipeline]`，并保留路径原样。`from_toml_str_strict`、`from_toml_file_strict`
仅启用严格解析。`from_toml_str_with_base` 可指定字符串配置的路径基准，
`resolve_model_paths(base)` 可为已有配置解析全部阶段的模型与字典路径。

`ConfigLoadOptions` 可分别设置 `ConfigParseMode` 和 `ModelPathBase`。路径策略包括
保留原样、工作目录、配置文件目录和指定目录；`ConfigLoadOptions::compatible()`
复现原入口的完整行为。解析后的路径为绝对路径，无需模型文件已存在；已有绝对路径保持
原样，允许 `..`。指定目录本身为相对路径时以加载时的工作目录为准。
Windows 的 `C:models`、`\models` 等路径在解析时必须补全驱动器。

CLI 默认严格解析，路径基准仍为工作目录，以兼容仓库现有配置。使用
`--config-compat` 恢复宽松解析与顶层覆盖行为，并在默认路径策略下保留路径原样。
使用 `--config-path-base config`
改为配置文件目录，或使用 `--config-base-dir <目录>` 指定基准。
以配置文件目录为基准时，`config/` 下的 TOML 应使用 `../models/...` 引用仓库模型。
