from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

RAPIDOCR_PYTHON_REPO_ENV = "RAPIDOCR_PYTHON_REPO"

try:
    import psutil
except ImportError:  # pragma: no cover - optional benchmark dependency
    psutil = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rs-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--python-repo",
        type=Path,
        default=None,
        help=f"Path to the Python RapidOCR repo. Defaults to ${RAPIDOCR_PYTHON_REPO_ENV}.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Shared Rust TOML config; model paths are relative to --rs-root. Defaults to ppocrv6-small.",
    )
    parser.add_argument(
        "--ort-dylib", type=Path, default=None,
        help="Rust ONNX Runtime library. Python's onnxruntime.dll can align the runtime version.",
    )
    parser.add_argument(
        "--rust-binary", type=Path, default=None,
        help="Use an existing release CLI instead of building it.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=None,
        help="Image path relative to the Python RapidOCR repo, or an absolute path.",
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sample-interval-ms", type=float, default=10.0)
    parser.add_argument("--no-cls", action="store_true")
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--python-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional Markdown file for recording the benchmark result.",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if not 0 < args.sample_interval_ms < float("inf"):
        parser.error("--sample-interval-ms must be finite and positive")
    return args


@dataclass
class BenchStats:
    name: str
    mean_ms: float
    min_ms: float
    max_ms: float
    total_ms: float | None = None
    model_load_ms: float | None = None
    peak_rss_bytes: int | None = None
    os_peak_rss_bytes: int | None = None
    peak_private_bytes: int | None = None
    post_call_rss_bytes: int | None = None
    line_count: int | None = None
    stages: dict[str, float] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class SampledMemory:
    peak_rss_bytes: int | None = None
    os_peak_rss_bytes: int | None = None
    peak_private_bytes: int | None = None

    def sample(self, process) -> None:
        info = memory_info(process)
        if info is None:
            return
        self.peak_rss_bytes = max_optional(self.peak_rss_bytes, int(info.rss))
        # Windows exposes the process high-water working set and Private Bytes.
        self.os_peak_rss_bytes = max_optional(
            self.os_peak_rss_bytes, getattr(info, "peak_wset", None)
        )
        self.peak_private_bytes = max_optional(
            self.peak_private_bytes, getattr(info, "private", None)
        )

    def apply(self, stats: BenchStats) -> None:
        stats.peak_rss_bytes = self.peak_rss_bytes
        stats.os_peak_rss_bytes = max_optional(
            stats.os_peak_rss_bytes, self.os_peak_rss_bytes
        )
        stats.peak_private_bytes = self.peak_private_bytes


def main() -> None:
    args = parse_args()
    rs_root = args.rs_root.resolve()
    python_repo = resolve_python_repo(args.python_repo)
    model_dir = resolve_under(rs_root, args.model_dir)
    repeat = max(args.repeat, 1)
    config_path = resolve_under(rs_root, args.config) if args.config else None
    config = load_benchmark_config(rs_root, model_dir, config_path)
    use_cls = config.get("pipeline", {}).get("use_cls", True) and not args.no_cls
    params = python_params(config, model_dir, use_cls, validate_assets=not args.skip_python)
    images = args.image or [Path("python/tests/test_files/ch_en_num.jpg")]

    if args.python_worker:
        with contextlib.redirect_stdout(sys.stderr):
            stats = bench_python_worker(
                python_repo, fixture_image_path(python_repo, images[0]),
                params, repeat, args.warmup, config.get("inference", {}),
            )
        print(json.dumps(asdict(stats)))
        return

    binary = None
    if not args.skip_rust:
        binary = build_rust_cli(rs_root, args.rust_binary, args.ort_dylib)

    rows = []
    for image in images:
        image_path = fixture_image_path(python_repo, image)
        image_display = display_path(python_repo, image_path)
        print(
            f"image={image_display} "
            f"repeat={repeat} warmup={args.warmup} use_cls={use_cls}"
        )
        if not args.skip_rust:
            rust_stats = bench_rust(
                rs_root, image_path, model_dir, repeat, not use_cls,
                binary, config_path, args.ort_dylib, args.warmup,
                args.sample_interval_ms / 1000.0,
            )
            print_stats(rust_stats)
            rows.append((image_display, use_cls, rust_stats))
        if not args.skip_python:
            python_stats = bench_python(
                rs_root, python_repo, image_path, model_dir, repeat, not use_cls,
                config_path, args.warmup, args.sample_interval_ms / 1000.0,
            )
            print_stats(python_stats)
            rows.append((image_display, use_cls, python_stats))
            if binary is not None and rust_stats.line_count != python_stats.line_count:
                print("warning: output line counts differ; timings do not establish OCR parity", file=sys.stderr)

    if args.out is not None:
        write_markdown_result(
            args.out, rs_root, python_repo, model_dir, repeat, rows,
            {
                "config": str(config_path) if config_path else "ppocrv6-small (default)",
                "warmup": args.warmup,
                "sample_interval_ms": args.sample_interval_ms,
                "rust_ort_dylib": str(args.ort_dylib) if args.ort_dylib else "bundled runtime",
                "inference": config.get("inference", {}),
                "models": {key: value for key, value in params.items() if key.endswith(("model_path", "rec_keys_path"))},
            },
        )


def resolve_python_repo(value: Path | None) -> Path:
    if value is None:
        env_value = os.environ.get(RAPIDOCR_PYTHON_REPO_ENV)
        if env_value:
            value = Path(env_value)
    if value is None:
        raise SystemExit(
            f"--python-repo or ${RAPIDOCR_PYTHON_REPO_ENV} is required for benchmarks"
        )

    repo = value.resolve()
    package_dir = repo / "python" / "rapidocr"
    if not package_dir.is_dir():
        raise SystemExit(f"{repo} does not look like the Python RapidOCR repo")
    return repo


def resolve_under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def fixture_image_path(python_repo: Path, image: Path) -> Path:
    return image.resolve() if image.is_absolute() else (python_repo / image).resolve()


def display_path(base: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_benchmark_config(rs_root: Path, model_dir: Path, path: Path | None) -> dict[str, Any]:
    source = path or rs_root / "config/ppocrv6-small.toml"
    with source.open("rb") as file:
        config = tomllib.load(file)
    if config.get("inference", {}).get("execution_provider", "cpu") != "cpu":
        raise SystemExit("This comparison currently supports CPU configs only")
    if not all(config.get("pipeline", {}).get(f"use_{stage}", True) for stage in ("det", "rec")):
        raise SystemExit("This comparison requires detection and recognition; classification is optional")
    for stage in ("det", "cls", "rec"):
        for key in ("model_path", "dict_path"):
            value = config.get(stage, {}).get(key)
            if value:
                resolved = resolve_under(rs_root, Path(value)) if path else model_dir / Path(value).name
                config[stage][key] = str(resolved)
    return config


def python_params(
    config: dict[str, Any], model_dir: Path, use_cls: bool, *, validate_assets: bool = True,
) -> dict[str, Any]:
    inference = config.get("inference", {})
    params = {
        "Global.model_root_dir": str(model_dir),
        "Global.log_level": "error",
        "Global.use_cls": use_cls,
        "EngineConfig.onnxruntime.intra_op_num_threads": inference.get("intra_threads", 1),
        "EngineConfig.onnxruntime.inter_op_num_threads": inference.get("inter_threads", 1),
        "EngineConfig.onnxruntime.enable_cpu_mem_arena": inference.get("enable_cpu_mem_arena", False),
    }
    for key in ("text_score", "min_side_len", "max_side_len", "min_height", "width_height_ratio"):
        if key in config:
            params[f"Global.{key}"] = config[key]
    mappings = {
        "det": {"model_path": "model_path", "limit_side_len": "limit_side_len", "limit_type": "limit_type",
                "mean": "mean", "std": "std", "thresh": "thresh", "box_thresh": "box_thresh",
                "max_candidates": "max_candidates", "unclip_ratio": "unclip_ratio"},
        "cls": {"model_path": "model_path", "image_shape": "cls_image_shape", "batch_size": "cls_batch_num",
                "thresh": "cls_thresh", "labels": "label_list"},
        "rec": {"model_path": "model_path", "dict_path": "rec_keys_path", "image_shape": "rec_img_shape",
                "batch_size": "rec_batch_num"},
    }
    for stage, mapping in mappings.items():
        for source, target in mapping.items():
            if source in config.get(stage, {}):
                value = config[stage][source]
                params[f"{stage.capitalize()}.{target}"] = value.lower() if source == "limit_type" else value
    # The reference initializes its classifier even when use_cls=False.
    params.setdefault("Cls.model_path", str(model_dir / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"))
    for key, value in params.items():
        if validate_assets and key.endswith(("model_path", "rec_keys_path")) and not Path(value).is_file():
            raise SystemExit(f"Missing benchmark asset for {key}: {value}")
    if validate_assets and config.get("rec", {}).get("return_tokens", False):
        raise SystemExit("Use return_tokens=false to compare the common text-only output")
    return params


def build_rust_cli(rs_root: Path, binary: Path | None, ort_dylib: Path | None) -> Path:
    if binary is not None:
        return resolve_under(rs_root, binary)
    cmd = ["cargo", "build", "--release", "--locked", "-q", "-p", "rapidocr-cli"]
    if ort_dylib is not None:
        cmd.extend(["--features", "load-dynamic"])
    subprocess.run(cmd, cwd=rs_root, check=True)
    return rs_root / "target" / "release" / ("rapidocr.exe" if sys.platform == "win32" else "rapidocr")


def bench_rust(
    rs_root: Path,
    image: Path,
    model_dir: Path,
    repeat: int,
    no_cls: bool,
    binary: Path,
    config_path: Path | None,
    ort_dylib: Path | None,
    warmup: int,
    sample_interval: float,
) -> BenchStats:
    cmd = [
        str(binary),
        "--image",
        str(image),
        "--model-dir",
        str(model_dir),
        "--no-download",
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
        "--quiet",
        "--benchmark-json",
    ]
    if no_cls:
        cmd.append("--no-cls")
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    if ort_dylib is not None:
        cmd.extend(["--ort-dylib", str(resolve_under(rs_root, ort_dylib))])

    stdout, stderr, memory = run_sampled(cmd, rs_root, sample_interval)
    parsed = parse_rust_json(stdout, memory.peak_rss_bytes)
    if parsed is not None:
        memory.apply(parsed)
        return parsed
    raise RuntimeError(f"Rust CLI did not emit valid benchmark JSON: {stderr}")


def bench_python(
    rs_root: Path,
    python_repo: Path,
    image: Path,
    model_dir: Path,
    repeat: int,
    no_cls: bool,
    config_path: Path | None,
    warmup: int,
    sample_interval: float,
) -> BenchStats:
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--python-worker",
        "--rs-root", str(rs_root), "--python-repo", str(python_repo),
        "--model-dir", str(model_dir), "--image", str(image),
        "--repeat", str(repeat), "--warmup", str(warmup),
    ]
    if no_cls:
        cmd.append("--no-cls")
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    stdout, _, memory = run_sampled(cmd, rs_root, sample_interval)
    stats = BenchStats(**json.loads(stdout))
    memory.apply(stats)
    return stats


def bench_python_worker(
    python_repo: Path,
    image: Path,
    params: dict[str, Any],
    repeat: int,
    warmup: int,
    inference: dict[str, Any],
) -> BenchStats:
    sys.path.insert(0, str(python_repo / "python"))

    from rapidocr import RapidOCR
    from rapidocr.inference_engine.onnxruntime.main import OrtInferSession
    import onnxruntime

    # Rust disables memory patterns for the dynamic OCR model inputs. Match that
    # setting without changing the reference repository or its installed package.
    original_options = OrtInferSession._init_sess_opts

    def session_options(cfg):
        options = original_options(cfg)
        # The reference ignores thread counts above os.cpu_count(); honor the
        # shared config directly, including intentional oversubscription.
        options.intra_op_num_threads = params["EngineConfig.onnxruntime.intra_op_num_threads"]
        options.inter_op_num_threads = params["EngineConfig.onnxruntime.inter_op_num_threads"]
        options.enable_mem_pattern = False
        options.execution_mode = (
            onnxruntime.ExecutionMode.ORT_PARALLEL if inference.get("parallel_execution", False)
            else onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        )
        return options

    OrtInferSession._init_sess_opts = staticmethod(session_options)

    process = psutil.Process(os.getpid()) if psutil is not None else None
    post_call_rss_bytes = current_rss(process)
    start = time.perf_counter()
    engine = RapidOCR(params=params)
    model_load_ms = (time.perf_counter() - start) * 1000.0
    post_call_rss_bytes = max_optional(post_call_rss_bytes, current_rss(process))
    for _ in range(warmup):
        engine(image)
        post_call_rss_bytes = max_optional(post_call_rss_bytes, current_rss(process))

    times = []
    stage_values: dict[str, list[float]] = {
        "det_ms": [],
        "cls_ms": [],
        "rec_ms": [],
    }
    for _ in range(repeat):
        start = time.perf_counter()
        result = engine(image)
        times.append((time.perf_counter() - start) * 1000.0)
        collect_python_stages(result, stage_values)
        post_call_rss_bytes = max_optional(post_call_rss_bytes, current_rss(process))
    stats = stats_from_values("python_hot", times)
    stats.model_load_ms = model_load_ms
    stats.post_call_rss_bytes = post_call_rss_bytes
    info = memory_info(process)
    stats.os_peak_rss_bytes = getattr(info, "peak_wset", None)
    stats.line_count = len(result.txts) if result.txts is not None else 0
    stats.metadata = {
        "onnxruntime": onnxruntime.__version__,
        "onnxruntime_build": onnxruntime.get_build_info(),
        "memory_pattern": False,
    }
    stats.stages = {
        name: statistics.mean(values) for name, values in stage_values.items() if values
    }
    return stats


def run_sampled(
    cmd: list[str], cwd: Path, sample_interval: float = 0.01,
) -> tuple[str, str, SampledMemory]:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        ps_process = psutil.Process(process.pid) if psutil is not None else None
    except psutil.NoSuchProcess:
        ps_process = None
    memory = SampledMemory()
    while True:
        memory.sample(ps_process)
        try:
            # communicate drains both pipes while waiting, including verbose
            # Python imports. Polling before draining can deadlock on a full pipe.
            stdout, stderr = process.communicate(timeout=sample_interval)
            break
        except subprocess.TimeoutExpired:
            pass
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
    return stdout, stderr, memory


def memory_info(process):
    if process is None:
        return None
    try:
        return process.memory_info()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def current_rss(process) -> int | None:
    info = memory_info(process)
    return int(info.rss) if info is not None else None


def max_optional(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def parse_rust_json(stdout: str, peak_rss_bytes: int | None) -> BenchStats | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None

    stages = rust_stage_summary(payload.get("mean_timings", {}))
    return BenchStats(
        name="rust_cli_hot",
        mean_ms=float(payload["mean_ms"]),
        min_ms=float(payload["min_ms"]),
        max_ms=float(payload["max_ms"]),
        total_ms=float(payload["total_ms"]),
        model_load_ms=float(payload["model_load_ms"]),
        peak_rss_bytes=peak_rss_bytes,
        line_count=payload.get("line_count"),
        stages=stages,
    )


def rust_stage_summary(timings: dict[str, Any]) -> dict[str, float]:
    def value(name: str) -> float:
        return float(timings.get(name, 0.0) or 0.0)

    return {
        "image_load_ms": value("image_load_ms"),
        "det_ms": value("pipeline_preprocess_ms")
        + value("det_preprocess_ms")
        + value("det_inference_ms")
        + value("det_postprocess_ms"),
        "det_inference_ms": value("det_inference_ms"),
        "det_postprocess_ms": value("det_postprocess_ms"),
        "crop_ms": value("crop_ms"),
        "cls_ms": value("cls_preprocess_ms")
        + value("cls_inference_ms")
        + value("cls_postprocess_ms"),
        "rec_ms": value("rec_preprocess_ms")
        + value("rec_inference_ms")
        + value("rec_decode_ms"),
        "output_filter_ms": value("output_filter_ms"),
    }


def collect_python_stages(result, stage_values: dict[str, list[float]]) -> None:
    elapse_list = getattr(result, "elapse_list", None)
    if elapse_list is not None and len(elapse_list) >= 3:
        append_stage_ms(stage_values, "det_ms", elapse_list[0])
        append_stage_ms(stage_values, "cls_ms", elapse_list[1])
        append_stage_ms(stage_values, "rec_ms", elapse_list[2])
        return

    elapse = getattr(result, "elapse", None)
    if elapse is not None:
        stage_values["rec_ms"].append(float(elapse) * 1000.0)


def append_stage_ms(stage_values: dict[str, list[float]], name: str, value) -> None:
    if value is None:
        return
    stage_values[name].append(float(value) * 1000.0)


def parse_rust_stats(stderr: str) -> BenchStats | None:
    values = {}
    for part in stderr.replace("\n", "\t").split("\t"):
        key, sep, value = part.partition("=")
        if sep and key in {"total_ms", "mean_ms", "min_ms", "max_ms"}:
            values[key] = float(value)
    if {"mean_ms", "min_ms", "max_ms"} <= values.keys():
        return BenchStats(
            name="rust_cli_hot",
            mean_ms=values["mean_ms"],
            min_ms=values["min_ms"],
            max_ms=values["max_ms"],
            total_ms=values.get("total_ms"),
        )
    return None


def parse_process_total_ms(stderr: str) -> float | None:
    for part in stderr.replace("\n", "\t").split("\t"):
        key, sep, value = part.partition("=")
        if sep and key == "total_ms":
            return float(value)
    return None


def stats_from_values(name: str, values: list[float]) -> BenchStats:
    return BenchStats(
        name=name,
        mean_ms=statistics.mean(values),
        min_ms=min(values),
        max_ms=max(values),
        total_ms=sum(values),
    )


def print_stats(stats: BenchStats) -> None:
    model_load = (
        f"\tmodel_load_ms={stats.model_load_ms:.3f}"
        if stats.model_load_ms is not None
        else ""
    )
    rss = "".join(
        f"\t{name}={value / (1024 * 1024):.1f}"
        for name, value in (
            ("peak_rss_mib", stats.peak_rss_bytes),
            ("os_peak_rss_mib", stats.os_peak_rss_bytes),
            ("peak_private_mib", stats.peak_private_bytes),
            ("post_call_rss_mib", stats.post_call_rss_bytes),
        )
        if value is not None
    )
    stages = format_stage_summary(stats)
    print(
        f"{stats.name}\tmean_ms={stats.mean_ms:.3f}\t"
        f"min_ms={stats.min_ms:.3f}\tmax_ms={stats.max_ms:.3f}"
        f"{model_load}{rss}{stages}\tline_count={stats.line_count}"
    )


def format_stage_summary(stats: BenchStats) -> str:
    if not stats.stages:
        return ""
    names = ["det_ms", "cls_ms", "rec_ms", "det_postprocess_ms"]
    parts = []
    for name in names:
        value = stats.stages.get(name)
        if value is not None:
            parts.append(f"{name}={value:.3f}")
    return "\t" + "\t".join(parts) if parts else ""


def write_markdown_result(
    path: Path,
    rs_root: Path,
    python_repo: Path,
    model_dir: Path,
    repeat: int,
    rows,
    metadata: dict[str, Any],
) -> None:
    path = resolve_under(rs_root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runner_metadata = {
        stats.name: stats.metadata for _, _, stats in rows if stats.metadata
    }
    lines = [
        "# Benchmark Result",
        "",
        f"- recorded_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- platform: {platform.platform()}",
        f"- python: {platform.python_version()}",
        f"- rust_repo: {rs_root}",
        f"- python_repo: {python_repo}",
        f"- model_dir: {model_dir}",
        f"- repeat: {repeat}",
        *(f"- {key}: {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items()),
        *(f"- {key}_metadata: {json.dumps(value, ensure_ascii=False)}" for key, value in runner_metadata.items()),
        "",
        "Both runners execute in fresh child processes and are sampled continuously, including imports, model load, warm-up, and inference. RSS units are MiB. On Windows, os_peak_rss_mib is the largest observed OS peak working set; peak_private_mib is sampled Private Bytes. Python post_call_rss_mib retains the old after-call-only reading for diagnosis and must not be compared with a continuously sampled peak.",
        "",
        "Python session memory patterns are disabled to match the dynamic Rust OCR inputs. The Python reference still initializes its classifier when classification is disabled. Rust input resource limits are not applied by Python; use inputs below those limits when comparing work.",
        "",
        "| image | use_cls | runner | model_load_ms | mean_ms | min_ms | max_ms | peak_rss_mib | os_peak_rss_mib | peak_private_mib | post_call_rss_mib | line_count | det_ms | cls_ms | rec_ms | det_postprocess_ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for image, use_cls, stats in rows:
        stages = stats.stages or {}
        memory = " | ".join(
            f"{value / (1024 * 1024):.1f}" if value is not None else ""
            for value in (stats.peak_rss_bytes, stats.os_peak_rss_bytes,
                          stats.peak_private_bytes, stats.post_call_rss_bytes)
        )
        model_load = f"{stats.model_load_ms:.3f}" if stats.model_load_ms is not None else ""
        lines.append(
            f"| `{image}` | {str(use_cls).lower()} | `{stats.name}` | "
            f"{model_load} | {stats.mean_ms:.3f} | {stats.min_ms:.3f} | {stats.max_ms:.3f} | "
            f"{memory} | {stats.line_count} | {format_stage_value(stages, 'det_ms')} | "
            f"{format_stage_value(stages, 'cls_ms')} | {format_stage_value(stages, 'rec_ms')} | "
            f"{format_stage_value(stages, 'det_postprocess_ms')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def format_stage_value(stages: dict[str, float], name: str) -> str:
    value = stages.get(name)
    return f"{value:.3f}" if value is not None else ""


if __name__ == "__main__":
    main()
