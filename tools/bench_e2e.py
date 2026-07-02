from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

RAPIDOCR_PYTHON_REPO_ENV = "RAPIDOCR_PYTHON_REPO"


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
    ]
    if no_cls:
        cmd.append("--no-cls")

    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=rs_root,
        check=True,
        text=True,
        capture_output=True,
    )
    total_ms = (time.perf_counter() - start) * 1000.0
    parsed = parse_rust_stats(completed.stderr)
    if parsed is not None:
        return parsed
    mean_ms = total_ms / repeat
    return BenchStats(
        name="rust_cli_hot",
        mean_ms=mean_ms,
        min_ms=mean_ms,
        max_ms=mean_ms,
        total_ms=total_ms,
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

    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(model_dir),
        }
    )
    use_cls = not no_cls
    engine(image, use_cls=use_cls)

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        engine(image, use_cls=use_cls)
        times.append((time.perf_counter() - start) * 1000.0)
    return stats_from_values("python_hot", times)


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


def stats_from_values(name: str, values: list[float]) -> BenchStats:
    return BenchStats(
        name=name,
        mean_ms=statistics.mean(values),
        min_ms=min(values),
        max_ms=max(values),
        total_ms=sum(values),
    )


def print_stats(stats: BenchStats) -> None:
    print(
        f"{stats.name}\tmean_ms={stats.mean_ms:.3f}\t"
        f"min_ms={stats.min_ms:.3f}\tmax_ms={stats.max_ms:.3f}"
    )


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
        "| image | use_cls | runner | mean_ms | min_ms | max_ms |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for image, use_cls, stats in rows:
        lines.append(
            f"| `{image}` | {str(use_cls).lower()} | `{stats.name}` | "
            f"{stats.mean_ms:.3f} | {stats.min_ms:.3f} | {stats.max_ms:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
