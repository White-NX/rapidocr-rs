from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time
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
        "--image",
        type=Path,
        action="append",
        default=None,
        help="Image path relative to the Python RapidOCR repo, or an absolute path.",
    )
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--no-cls", action="store_true")
    parser.add_argument("--skip-rust", action="store_true")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional Markdown file for recording the benchmark result.",
    )
    return parser.parse_args()


@dataclass
class BenchStats:
    name: str
    mean_ms: float
    min_ms: float
    max_ms: float
    total_ms: float | None = None
    model_load_ms: float | None = None
    peak_rss_bytes: int | None = None
    stages: dict[str, float] | None = None


def main() -> None:
    args = parse_args()
    rs_root = args.rs_root.resolve()
    python_repo = resolve_python_repo(args.python_repo)
    model_dir = resolve_under(rs_root, args.model_dir)
    repeat = max(args.repeat, 1)

    rows = []
    images = args.image or [Path("python/tests/test_files/ch_en_num.jpg")]
    for image in images:
        image_path = fixture_image_path(python_repo, image)
        image_display = display_path(python_repo, image_path)
        print(
            f"image={image_display} "
            f"repeat={repeat} use_cls={not args.no_cls}"
        )
        if not args.skip_rust:
            rust_stats = bench_rust(rs_root, image_path, model_dir, repeat, args.no_cls)
            print_stats(rust_stats)
            rows.append((image_display, not args.no_cls, rust_stats))
        if not args.skip_python:
            python_stats = bench_python(python_repo, image_path, model_dir, repeat, args.no_cls)
            print_stats(python_stats)
            rows.append((image_display, not args.no_cls, python_stats))

    if args.out is not None:
        write_markdown_result(args.out, rs_root, python_repo, model_dir, repeat, rows)


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


def bench_rust(
    rs_root: Path,
    image: Path,
    model_dir: Path,
    repeat: int,
    no_cls: bool,
) -> BenchStats:
    subprocess.run(
        ["cargo", "build", "--release", "-q", "-p", "rapidocr-cli"],
        cwd=rs_root,
        check=True,
    )
    binary = rs_root / "target" / "release" / ("rapidocr.exe" if sys.platform == "win32" else "rapidocr")
    cmd = [
        str(binary),
        "--image",
        str(image),
        "--model-dir",
        str(model_dir),
        "--no-download",
        "--repeat",
        str(repeat),
        "--quiet",
        "--benchmark-json",
    ]
    if no_cls:
        cmd.append("--no-cls")

    stdout, stderr, peak_rss_bytes = run_sampled(cmd, rs_root)
    parsed = parse_rust_json(stdout, peak_rss_bytes)
    if parsed is not None:
        return parsed

    total_ms = parse_process_total_ms(stderr)
    if total_ms is None:
        total_ms = 0.0
    mean_ms = total_ms / repeat
    return BenchStats(
        name="rust_cli_hot",
        mean_ms=mean_ms,
        min_ms=mean_ms,
        max_ms=mean_ms,
        total_ms=total_ms,
        peak_rss_bytes=peak_rss_bytes,
    )


def bench_python(
    python_repo: Path,
    image: Path,
    model_dir: Path,
    repeat: int,
    no_cls: bool,
) -> BenchStats:
    sys.path.insert(0, str(python_repo / "python"))

    from rapidocr import RapidOCR

    process = psutil.Process(os.getpid()) if psutil is not None else None
    peak_rss_bytes = current_rss(process)
    start = time.perf_counter()
    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(model_dir),
        }
    )
    model_load_ms = (time.perf_counter() - start) * 1000.0
    peak_rss_bytes = max_optional(peak_rss_bytes, current_rss(process))
    use_cls = not no_cls
    engine(image, use_cls=use_cls)
    peak_rss_bytes = max_optional(peak_rss_bytes, current_rss(process))

    times = []
    stage_values: dict[str, list[float]] = {
        "det_ms": [],
        "cls_ms": [],
        "rec_ms": [],
    }
    for _ in range(repeat):
        start = time.perf_counter()
        result = engine(image, use_cls=use_cls)
        times.append((time.perf_counter() - start) * 1000.0)
        collect_python_stages(result, stage_values)
        peak_rss_bytes = max_optional(peak_rss_bytes, current_rss(process))
    stats = stats_from_values("python_hot", times)
    stats.model_load_ms = model_load_ms
    stats.peak_rss_bytes = peak_rss_bytes
    stats.stages = {
        name: statistics.mean(values) for name, values in stage_values.items() if values
    }
    return stats


def run_sampled(cmd: list[str], cwd: Path) -> tuple[str, str, int | None]:
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    ps_process = psutil.Process(process.pid) if psutil is not None else None
    peak_rss_bytes = current_rss(ps_process)
    while process.poll() is None:
        peak_rss_bytes = max_optional(peak_rss_bytes, current_rss(ps_process))
        time.sleep(0.01)
    stdout, stderr = process.communicate()
    peak_rss_bytes = max_optional(peak_rss_bytes, current_rss(ps_process))
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
    return stdout, stderr, peak_rss_bytes


def current_rss(process) -> int | None:
    if process is None:
        return None
    try:
        return int(process.memory_info().rss)
    except Exception:
        return None


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
    rss = (
        f"\tpeak_rss_mb={stats.peak_rss_bytes / (1024 * 1024):.1f}"
        if stats.peak_rss_bytes is not None
        else ""
    )
    stages = format_stage_summary(stats)
    print(
        f"{stats.name}\tmean_ms={stats.mean_ms:.3f}\t"
        f"min_ms={stats.min_ms:.3f}\tmax_ms={stats.max_ms:.3f}"
        f"{model_load}{rss}{stages}"
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
) -> None:
    path = resolve_under(rs_root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "",
        "| image | use_cls | runner | model_load_ms | mean_ms | min_ms | max_ms | peak_rss_mb | det_ms | cls_ms | rec_ms | det_postprocess_ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for image, use_cls, stats in rows:
        stages = stats.stages or {}
        rss_mb = (
            f"{stats.peak_rss_bytes / (1024 * 1024):.1f}"
            if stats.peak_rss_bytes is not None
            else ""
        )
        model_load = f"{stats.model_load_ms:.3f}" if stats.model_load_ms is not None else ""
        lines.append(
            f"| `{image}` | {str(use_cls).lower()} | `{stats.name}` | "
            f"{model_load} | {stats.mean_ms:.3f} | {stats.min_ms:.3f} | {stats.max_ms:.3f} | "
            f"{rss_mb} | {format_stage_value(stages, 'det_ms')} | "
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
