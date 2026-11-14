use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use clap::{Parser, ValueEnum};
use rapidocr_core::{
    config::{
        ConfigLoadOptions, ConfigParseMode, ExecutionProvider, ModelPathBase, RapidOcrConfig,
    },
    model::{
        available_model_set_names, model_set_by_name, ModelCache, ModelDownloadMode, ModelSetSpec,
        DEFAULT_MODEL_SET_NAME,
    },
    types::OcrTimings,
    RapidOcr,
};
use serde::Serialize;

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long)]
    image: Option<PathBuf>,

    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Accept unknown TOML fields and let legacy top-level flags override [pipeline].
    #[arg(long, requires = "config")]
    config_compat: bool,

    /// Base for relative model/dictionary paths in --config (cwd preserves existing configs).
    #[arg(long, value_enum, default_value_t = ConfigPathBase::Cwd, conflicts_with = "config_base_dir")]
    config_path_base: ConfigPathBase,

    /// Explicit base directory for model/dictionary paths in --config.
    #[arg(long, requires = "config")]
    config_base_dir: Option<PathBuf>,

    #[arg(long, default_value = "models")]
    model_dir: PathBuf,

    #[arg(long, default_value = DEFAULT_MODEL_SET_NAME)]
    model_set: String,

    #[arg(long)]
    write_default_config: Option<PathBuf>,

    #[arg(long)]
    no_download: bool,

    #[arg(long)]
    directml: bool,

    /// Path to an application-provided ONNX Runtime library (requires load-dynamic).
    #[arg(long)]
    ort_dylib: Option<PathBuf>,

    #[arg(long)]
    no_det: bool,

    #[arg(long)]
    no_cls: bool,

    #[arg(long)]
    no_rec: bool,

    #[arg(long, default_value_t = 1)]
    repeat: usize,

    #[arg(long)]
    quiet: bool,

    #[arg(long)]
    benchmark_json: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ConfigPathBase {
    Cwd,
    Config,
}

fn load_config(path: &std::path::Path, args: &Args) -> rapidocr_core::OcrResult<RapidOcrConfig> {
    let parse_mode = if args.config_compat {
        ConfigParseMode::Compatible
    } else {
        ConfigParseMode::Strict
    };
    let path_base = match &args.config_base_dir {
        Some(base) => ModelPathBase::Directory(base.clone()),
        None => match args.config_path_base {
            ConfigPathBase::Cwd if args.config_compat => ModelPathBase::AsProvided,
            ConfigPathBase::Cwd => ModelPathBase::WorkingDirectory,
            ConfigPathBase::Config => ModelPathBase::ConfigDirectory,
        },
    };
    RapidOcrConfig::from_toml_file_with_options(
        path,
        &ConfigLoadOptions::new()
            .with_parse_mode(parse_mode)
            .with_model_path_base(path_base),
    )
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(path) = &args.write_default_config {
        let model_set = select_model_set(&args.model_set)?;
        let cache = ModelCache::new(&args.model_dir);
        let mut cfg = cache.config_for(model_set);
        apply_overrides(&mut cfg, &args);
        cfg.write_toml_file(path)?;
        return Ok(());
    }

    let image = args.image.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--image is required unless --write-default-config is used")
    })?;

    let mut cfg = if let Some(config_path) = &args.config {
        load_config(config_path, &args)?
    } else {
        let cache = ModelCache::new(&args.model_dir);
        let mode = if args.no_download {
            ModelDownloadMode::Never
        } else {
            ModelDownloadMode::Missing
        };
        let model_set = select_model_set(&args.model_set)?;
        let mut cfg = cache.config_for(model_set);
        apply_overrides(&mut cfg, &args);
        cfg.validate()?;
        cache.ensure_model_set_for_pipeline(model_set, cfg.pipeline, mode)?;
        cfg
    };

    apply_overrides(&mut cfg, &args);
    cfg.validate()?;
    let model_load_start = Instant::now();
    let mut ocr = if let Some(path) = &args.ort_dylib {
        #[cfg(feature = "load-dynamic")]
        {
            let runtime = rapidocr_core::OcrRuntime::load_from(path)?;
            RapidOcr::new_with_runtime(cfg, &runtime)?
        }
        #[cfg(not(feature = "load-dynamic"))]
        {
            let _ = path;
            anyhow::bail!("--ort-dylib requires the `load-dynamic` Cargo feature");
        }
    } else {
        RapidOcr::new(cfg)?
    };
    let model_load_ms = model_load_start.elapsed().as_secs_f64() * 1000.0;
    let repeat = args.repeat.max(1);
    let mut elapsed = Vec::with_capacity(repeat);
    let mut timings = Vec::with_capacity(repeat);
    let mut output = None;
    for _ in 0..repeat {
        let start = Instant::now();
        if args.benchmark_json {
            let timed = ocr.run_path_timed(image)?;
            output = Some(timed.output);
            timings.push(timed.timings);
        } else {
            output = Some(ocr.run_path(image)?);
        }
        elapsed.push(start.elapsed());
    }

    let elapsed_ms = elapsed
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    let total_ms = elapsed_ms.iter().sum::<f64>();
    let mean_ms = total_ms / repeat as f64;
    let min_ms = elapsed_ms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = elapsed_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if args.benchmark_json {
        let summary = BenchmarkJson {
            repeat,
            model_load_ms,
            total_ms,
            mean_ms,
            min_ms,
            max_ms,
            mean_timings: mean_timings(&timings),
            line_count: output.as_ref().map(|out| out.lines.len()).unwrap_or(0),
        };
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }

    if args.quiet || repeat > 1 {
        eprintln!(
            "repeat={repeat}\ttotal_ms={total_ms:.3}\tmean_ms={mean_ms:.3}\tmin_ms={min_ms:.3}\tmax_ms={max_ms:.3}"
        );
    }

    if !args.quiet {
        for line in output.unwrap().lines {
            print!("{:.5}\t{}\t{:?}", line.score, line.text, line.bbox.points);
            if let Some(score) = line.detection_score {
                print!("\tdet_score={score:.5}");
            }
            if let Some(rotation) = line.rotation {
                print!(
                    "\tcls_label={:?}\tcls_score={:.5}\trotation={}deg",
                    rotation.label, rotation.score, rotation.applied_degrees
                );
            }
            println!();
        }
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct BenchmarkJson {
    repeat: usize,
    model_load_ms: f64,
    total_ms: f64,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    mean_timings: OcrTimings,
    line_count: usize,
}

fn mean_timings(timings: &[OcrTimings]) -> OcrTimings {
    let mut out = OcrTimings::default();
    for timing in timings {
        out.add_assign(timing);
    }
    out.div_by(timings.len().max(1) as f64)
}

fn select_model_set(name: &str) -> Result<&'static ModelSetSpec> {
    model_set_by_name(name).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown model set {name:?}; available model sets: {}",
            available_model_set_names().join(", ")
        )
    })
}

fn apply_overrides(cfg: &mut RapidOcrConfig, args: &Args) {
    if args.no_det {
        cfg.pipeline.use_det = false;
    }
    if args.no_rec {
        cfg.pipeline.use_rec = false;
        cfg.pipeline.use_cls = false;
    }
    if args.no_cls {
        cfg.pipeline.use_cls = false;
    }
    if args.directml {
        cfg.inference.execution_provider = ExecutionProvider::DirectMl;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        io::Write,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_CONFIG: AtomicU64 = AtomicU64::new(0);

    struct TestConfig(PathBuf);

    impl TestConfig {
        fn new(prefix: &str) -> Self {
            let text = format!(
                "{prefix}{}",
                RapidOcrConfig::ppocr_v6_small("models")
                    .to_toml_string()
                    .unwrap()
            );
            loop {
                let id = NEXT_CONFIG.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "rapidocr-cli-config-{}-{id}.toml",
                    std::process::id()
                ));
                match fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                {
                    Ok(mut file) => {
                        file.write_all(text.as_bytes()).unwrap();
                        return Self(path);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(error) => panic!("failed to create config fixture: {error}"),
                }
            }
        }
    }

    impl Drop for TestConfig {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    #[test]
    fn config_is_strict_by_default_and_compat_flag_restores_legacy_overrides() {
        let config = TestConfig::new("extension = true\nuse_det = false\n");
        let path = config.0.to_str().unwrap();
        let strict = Args::try_parse_from(["rapidocr", "--config", path]).unwrap();
        let error = load_config(&config.0, &strict).unwrap_err();
        assert!(matches!(
            error,
            rapidocr_core::OcrError::Config {
                kind: rapidocr_core::ConfigErrorKind::UnknownField,
                ..
            }
        ));
        let compat =
            Args::try_parse_from(["rapidocr", "--config", path, "--config-compat"]).unwrap();
        assert!(!load_config(&config.0, &compat).unwrap().pipeline.use_det);
    }

    #[test]
    fn config_compat_preserves_paths_and_unused_empty_stage_paths() {
        let config = TestConfig::new("use_det = false\n");
        let mut parsed = RapidOcrConfig::from_toml_file(&config.0).unwrap();
        parsed.det.as_mut().unwrap().model_path = PathBuf::new();
        parsed.write_toml_file(&config.0).unwrap();
        let path = config.0.to_str().unwrap();
        let compat =
            Args::try_parse_from(["rapidocr", "--config", path, "--config-compat"]).unwrap();
        let loaded = load_config(&config.0, &compat).unwrap();
        assert!(loaded.det.unwrap().model_path.as_os_str().is_empty());
        assert_eq!(
            loaded.rec.unwrap().dict_path,
            PathBuf::from("models/ppocrv6_dict.txt")
        );
    }

    #[test]
    fn config_path_flags_select_cwd_config_directory_and_an_explicit_base() {
        let config = TestConfig::new("");
        let path = config.0.to_str().unwrap();
        let cwd = Args::try_parse_from(["rapidocr", "--config", path]).unwrap();
        assert_eq!(
            load_config(&config.0, &cwd).unwrap().rec.unwrap().dict_path,
            std::env::current_dir()
                .unwrap()
                .join("models/ppocrv6_dict.txt")
        );
        let relative =
            Args::try_parse_from(["rapidocr", "--config", path, "--config-path-base", "config"])
                .unwrap();
        assert_eq!(
            load_config(&config.0, &relative)
                .unwrap()
                .rec
                .unwrap()
                .dict_path,
            config.0.parent().unwrap().join("models/ppocrv6_dict.txt")
        );
        let explicit = Args::try_parse_from([
            "rapidocr",
            "--config",
            path,
            "--config-base-dir",
            "deployment",
        ])
        .unwrap();
        assert_eq!(
            load_config(&config.0, &explicit)
                .unwrap()
                .rec
                .unwrap()
                .dict_path,
            std::env::current_dir()
                .unwrap()
                .join("deployment/models/ppocrv6_dict.txt")
        );
        assert!(Args::try_parse_from([
            "rapidocr",
            "--config",
            path,
            "--config-path-base",
            "config",
            "--config-base-dir",
            "deployment"
        ])
        .is_err());
    }

    #[test]
    fn select_model_set_accepts_default_model_set() {
        let model_set = select_model_set(DEFAULT_MODEL_SET_NAME).unwrap();

        assert_eq!(model_set.name, DEFAULT_MODEL_SET_NAME);
    }

    #[test]
    fn select_model_set_reports_available_names_for_unknown_model_set() {
        let err = select_model_set("missing-model-set").unwrap_err();
        let message = err.to_string();

        assert!(message.contains("unknown model set"));
        assert!(message.contains("missing-model-set"));
        assert!(message.contains(DEFAULT_MODEL_SET_NAME));
    }

    #[test]
    fn directml_flag_selects_directml_execution_provider() {
        let args = Args::try_parse_from(["rapidocr", "--directml"]).unwrap();
        let mut cfg = RapidOcrConfig::ppocr_v6_small("models");

        apply_overrides(&mut cfg, &args);

        assert_eq!(
            cfg.inference.execution_provider,
            ExecutionProvider::DirectMl
        );
    }
}
