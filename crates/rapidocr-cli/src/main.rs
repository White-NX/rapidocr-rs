use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use clap::Parser;
use rapidocr_core::{
    config::RapidOcrConfig,
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

    #[arg(long, default_value = "models")]
    model_dir: PathBuf,

    #[arg(long, default_value = DEFAULT_MODEL_SET_NAME)]
    model_set: String,

    #[arg(long)]
    write_default_config: Option<PathBuf>,

    #[arg(long)]
    no_download: bool,

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

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(path) = &args.write_default_config {
        let model_set = select_model_set(&args.model_set)?;
        let cache = ModelCache::new(&args.model_dir);
        let mut cfg = cache.config_for(model_set);
        apply_pipeline_overrides(&mut cfg, &args);
        cfg.write_toml_file(path)?;
        return Ok(());
    }

    let image = args.image.as_ref().ok_or_else(|| {
        anyhow::anyhow!("--image is required unless --write-default-config is used")
    })?;

    let mut cfg = if let Some(config_path) = &args.config {
        RapidOcrConfig::from_toml_file(config_path)?
    } else {
        let cache = ModelCache::new(&args.model_dir);
        let mode = if args.no_download {
            ModelDownloadMode::Never
        } else {
            ModelDownloadMode::Missing
        };
        let model_set = select_model_set(&args.model_set)?;
        let mut cfg = cache.config_for(model_set);
        apply_pipeline_overrides(&mut cfg, &args);
        cfg.validate()?;
        cache.ensure_model_set_for_pipeline(model_set, cfg.pipeline, mode)?;
        cfg
    };

    apply_pipeline_overrides(&mut cfg, &args);
    cfg.validate()?;
    let model_load_start = Instant::now();
    let mut ocr = RapidOcr::new(cfg)?;
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
            println!("{:.5}\t{}\t{:?}", line.score, line.text, line.bbox.points);
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

fn apply_pipeline_overrides(cfg: &mut RapidOcrConfig, args: &Args) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
