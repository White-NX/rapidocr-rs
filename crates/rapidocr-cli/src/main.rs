use std::{path::PathBuf, time::Instant};

use anyhow::Result;
use clap::Parser;
use rapidocr_core::{
    config::RapidOcrConfig,
    model::{ModelCache, ModelDownloadMode, PPOCRV6_SMALL},
    RapidOcr,
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long)]
    image: Option<PathBuf>,

    #[arg(short, long)]
    config: Option<PathBuf>,

    #[arg(long, default_value = "models")]
    model_dir: PathBuf,

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
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(path) = &args.write_default_config {
        let cache = ModelCache::new(&args.model_dir);
        let mut cfg = cache.config_for(&PPOCRV6_SMALL);
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
        let mut cfg = cache.config_for(&PPOCRV6_SMALL);
        apply_pipeline_overrides(&mut cfg, &args);
        cfg.validate()?;
        cache.ensure_model_set_for_pipeline(&PPOCRV6_SMALL, cfg.pipeline, mode)?;
        cfg
    };

    apply_pipeline_overrides(&mut cfg, &args);
    cfg.validate()?;
    let mut ocr = RapidOcr::new(cfg)?;
    let repeat = args.repeat.max(1);
    let mut elapsed = Vec::with_capacity(repeat);
    let mut output = None;
    for _ in 0..repeat {
        let start = Instant::now();
        output = Some(ocr.run_path(image)?);
        elapsed.push(start.elapsed());
    }

    if args.quiet || repeat > 1 {
        let elapsed_ms = elapsed
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect::<Vec<_>>();
        let total_ms = elapsed_ms.iter().sum::<f64>();
        let mean_ms = total_ms / repeat as f64;
        let min_ms = elapsed_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let max_ms = elapsed_ms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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
