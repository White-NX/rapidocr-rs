from __future__ import annotations

import argparse
import json
import os
import sys
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
        "--out-dir",
        type=Path,
        default=Path("fixtures/e2e"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rs_root = args.rs_root.resolve()
    python_repo = resolve_python_repo(args.python_repo)
    python_dir = python_repo / "python"
    sys.path.insert(0, str(python_dir))

    from rapidocr import RapidOCR

    model_dir = resolve_under(rs_root, args.model_dir)
    out_dir = resolve_under(rs_root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(model_dir),
        }
    )

    cases = [
        ("ch_en_num_cls", Path("python/tests/test_files/ch_en_num.jpg"), True, None),
        ("ch_en_num_no_cls", Path("python/tests/test_files/ch_en_num.jpg"), False, None),
        ("text_det_cls", Path("python/tests/test_files/text_det.jpg"), True, None),
        ("text_det_no_cls", Path("python/tests/test_files/text_det.jpg"), False, None),
        ("en_cls", Path("python/tests/test_files/en.jpg"), True, None),
        ("en_no_cls", Path("python/tests/test_files/en.jpg"), False, None),
        ("empty_black_cls", Path("python/tests/test_files/empty_black.jpg"), True, None),
        ("empty_black_no_cls", Path("python/tests/test_files/empty_black.jpg"), False, None),
        ("test_letterbox_like_cls", Path("python/tests/test_files/test_letterbox_like.jpg"), True, None),
        ("test_letterbox_like_no_cls", Path("python/tests/test_files/test_letterbox_like.jpg"), False, None),
        ("text_vertical_words_cls", Path("python/tests/test_files/text_vertical_words.png"), True, None),
        ("text_vertical_words_no_cls", Path("python/tests/test_files/text_vertical_words.png"), False, None),
        ("latin_cls", Path("python/tests/test_files/latin.jpg"), True, None),
        ("latin_no_cls", Path("python/tests/test_files/latin.jpg"), False, None),
        (
            "issue_170_cls",
            Path("python/tests/test_files/issue_170.png"),
            True,
            {"max_mean_corner_delta": 8.0},
        ),
        (
            "issue_170_no_cls",
            Path("python/tests/test_files/issue_170.png"),
            False,
            {"max_mean_corner_delta": 8.0},
        ),
    ]

    for name, image, use_cls, tolerances in cases:
        image_path = fixture_image_path(python_repo, image)
        result = engine(image_path, use_cls=use_cls)
        export_one(
            out_dir / f"{name}.json",
            display_path(python_repo, image_path),
            use_cls,
            result,
            tolerances,
        )


def resolve_python_repo(value: Path | None) -> Path:
    if value is None:
        env_value = os.environ.get(RAPIDOCR_PYTHON_REPO_ENV)
        if env_value:
            value = Path(env_value)
    if value is None:
        raise SystemExit(
            f"--python-repo or ${RAPIDOCR_PYTHON_REPO_ENV} is required for parity export"
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


def export_one(path: Path, image: str, use_cls: bool, result, tolerances=None) -> None:
    boxes = result.boxes.tolist() if result.boxes is not None else []
    txts = list(result.txts or [])
    scores = list(result.scores or [])
    lines = []
    for box, text, score in zip(boxes, txts, scores):
        lines.append(
            {
                "bbox": [[float(x), float(y)] for x, y in box],
                "text": text,
                "score": float(score),
            }
        )

    payload = {
        "source": "python-rapidocr",
        "image": image,
        "use_cls": use_cls,
        "lines": lines,
    }
    if tolerances:
        payload["tolerances"] = tolerances
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
