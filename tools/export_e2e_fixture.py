from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path

RAPIDOCR_PYTHON_REPO_ENV = "RAPIDOCR_PYTHON_REPO"


@dataclass(frozen=True)
class E2eCase:
    name: str
    image: Path
    pipeline: dict[str, bool]
    tolerances: dict[str, float] | None = None


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
        help="Image path relative to the Python RapidOCR repo, or an absolute path. Custom full/rec-only images export cls and no-cls fixtures; det-only exports one fixture.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["full", "rec-only", "det-only"],
        default="full",
        help="Pipeline used with custom --image exports.",
    )
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

    cases = (
        custom_cases(args.image, python_repo, args.pipeline)
        if args.image
        else default_cases()
    )
    for case in cases:
        image_path = fixture_image_path(python_repo, case.image)
        result = engine(image_path, **case.pipeline)
        export_one(
            out_dir / f"{case.name}.json",
            display_path(python_repo, image_path),
            case.pipeline,
            result,
            case.tolerances,
        )


def full_pipeline(use_cls: bool) -> dict[str, bool]:
    return {"use_det": True, "use_cls": use_cls, "use_rec": True}


def rec_only_pipeline(use_cls: bool) -> dict[str, bool]:
    return {"use_det": False, "use_cls": use_cls, "use_rec": True}


def det_only_pipeline() -> dict[str, bool]:
    return {"use_det": True, "use_cls": False, "use_rec": False}


def custom_cases(images: list[Path], python_repo: Path, pipeline: str) -> list[E2eCase]:
    cases = []
    for image in images:
        image_path = fixture_image_path(python_repo, image)
        name = image_path.stem.replace("-", "_")
        display = Path(display_path(python_repo, image_path))
        if pipeline == "det-only":
            cases.append(E2eCase(f"{name}_det_only", display, det_only_pipeline()))
            continue

        pipeline_fn = rec_only_pipeline if pipeline == "rec-only" else full_pipeline
        suffix = "rec_only_" if pipeline == "rec-only" else ""
        cases.append(E2eCase(f"{name}_{suffix}cls", display, pipeline_fn(True)))
        cases.append(E2eCase(f"{name}_{suffix}no_cls", display, pipeline_fn(False)))
    return cases


def default_cases() -> list[E2eCase]:
    return [
        E2eCase("ch_en_num_cls", Path("python/tests/test_files/ch_en_num.jpg"), full_pipeline(True)),
        E2eCase("ch_en_num_no_cls", Path("python/tests/test_files/ch_en_num.jpg"), full_pipeline(False)),
        E2eCase("ch_en_num_det_only", Path("python/tests/test_files/ch_en_num.jpg"), det_only_pipeline()),
        E2eCase("text_det_cls", Path("python/tests/test_files/text_det.jpg"), full_pipeline(True)),
        E2eCase("text_det_no_cls", Path("python/tests/test_files/text_det.jpg"), full_pipeline(False)),
        E2eCase("text_det_det_only", Path("python/tests/test_files/text_det.jpg"), det_only_pipeline()),
        E2eCase("check_return_word_len_det_only", Path("python/tests/test_files/check_return_word_len.jpeg"), det_only_pipeline()),
        E2eCase("arabic_det_only", Path("python/tests/test_files/arabic.png"), det_only_pipeline()),
        E2eCase("cyrillic_det_only", Path("python/tests/test_files/cyrillic.png"), det_only_pipeline()),
        E2eCase("devanagari_det_only", Path("python/tests/test_files/devanagari.jpg"), det_only_pipeline()),
        E2eCase("japan_det_only", Path("python/tests/test_files/japan.jpg"), det_only_pipeline()),
        E2eCase("korean_det_only", Path("python/tests/test_files/korean.jpg"), det_only_pipeline()),
        E2eCase("en_cls", Path("python/tests/test_files/en.jpg"), full_pipeline(True)),
        E2eCase("en_no_cls", Path("python/tests/test_files/en.jpg"), full_pipeline(False)),
        E2eCase("empty_black_cls", Path("python/tests/test_files/empty_black.jpg"), full_pipeline(True)),
        E2eCase("empty_black_no_cls", Path("python/tests/test_files/empty_black.jpg"), full_pipeline(False)),
        E2eCase("short_cls", Path("python/tests/test_files/short.png"), full_pipeline(True)),
        E2eCase("short_no_cls", Path("python/tests/test_files/short.png"), full_pipeline(False)),
        E2eCase("test_letterbox_like_cls", Path("python/tests/test_files/test_letterbox_like.jpg"), full_pipeline(True)),
        E2eCase("test_letterbox_like_no_cls", Path("python/tests/test_files/test_letterbox_like.jpg"), full_pipeline(False)),
        E2eCase("test_without_det_cls", Path("python/tests/test_files/test_without_det.jpg"), full_pipeline(True)),
        E2eCase("test_without_det_no_cls", Path("python/tests/test_files/test_without_det.jpg"), full_pipeline(False)),
        E2eCase("img_exif_orientation_cls", Path("python/tests/test_files/img_exif_orientation.jpg"), full_pipeline(True)),
        E2eCase("img_exif_orientation_no_cls", Path("python/tests/test_files/img_exif_orientation.jpg"), full_pipeline(False)),
        E2eCase("black_font_color_transparent_cls", Path("python/tests/test_files/black_font_color_transparent.png"), full_pipeline(True)),
        E2eCase("black_font_color_transparent_no_cls", Path("python/tests/test_files/black_font_color_transparent.png"), full_pipeline(False)),
        E2eCase("ch_doc_server_cls", Path("python/tests/test_files/ch_doc_server.png"), full_pipeline(True)),
        E2eCase("return_word_debug_cls", Path("python/tests/test_files/return_word_debug.jpg"), full_pipeline(True)),
        E2eCase("text_vertical_words_cls", Path("python/tests/test_files/text_vertical_words.png"), full_pipeline(True)),
        E2eCase("text_vertical_words_no_cls", Path("python/tests/test_files/text_vertical_words.png"), full_pipeline(False)),
        E2eCase("latin_cls", Path("python/tests/test_files/latin.jpg"), full_pipeline(True)),
        E2eCase("latin_no_cls", Path("python/tests/test_files/latin.jpg"), full_pipeline(False)),
        E2eCase(
            "issue_170_cls",
            Path("python/tests/test_files/issue_170.png"),
            full_pipeline(True),
            {"max_mean_corner_delta": 8.0},
        ),
        E2eCase(
            "issue_170_no_cls",
            Path("python/tests/test_files/issue_170.png"),
            full_pipeline(False),
            {"max_mean_corner_delta": 8.0},
        ),
        E2eCase("en_rec_rec_only_cls", Path("python/tests/test_files/en_rec.jpg"), rec_only_pipeline(True)),
        E2eCase("en_rec_rec_only_no_cls", Path("python/tests/test_files/en_rec.jpg"), rec_only_pipeline(False)),
        E2eCase("text_rec_rec_only_cls", Path("python/tests/test_files/text_rec.jpg"), rec_only_pipeline(True)),
        E2eCase("text_rec_rec_only_no_cls", Path("python/tests/test_files/text_rec.jpg"), rec_only_pipeline(False)),
        E2eCase("text_cls_rec_only_cls", Path("python/tests/test_files/text_cls.jpg"), rec_only_pipeline(True)),
        E2eCase("text_cls_rec_only_no_cls", Path("python/tests/test_files/text_cls.jpg"), rec_only_pipeline(False)),
    ]


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


def export_one(path: Path, image: str, pipeline: dict[str, bool], result, tolerances=None) -> None:
    lines = []
    if pipeline["use_rec"]:
        txts = list(result.txts or [])
        scores = list(result.scores or [])
        boxes = result_boxes(result, len(txts))
        for box, text, score in zip(boxes, txts, scores):
            if not str(text).strip():
                continue
            lines.append(
                {
                    "bbox": [[float(x), float(y)] for x, y in box],
                    "text": text,
                    "score": float(score),
                }
            )
    else:
        boxes = result_boxes(result, 0)
        for box in boxes:
            lines.append(
                {
                    "bbox": [[float(x), float(y)] for x, y in box],
                    "text": "",
                    "score": 0.0,
                }
            )

    payload = {
        "source": "python-rapidocr",
        "image": image,
        "use_cls": pipeline["use_cls"],
        "lines": lines,
    }
    if pipeline != full_pipeline(pipeline["use_cls"]):
        payload["pipeline"] = pipeline
    if tolerances:
        payload["tolerances"] = tolerances
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def result_boxes(result, line_count: int) -> list:
    if hasattr(result, "boxes") and result.boxes is not None:
        return result.boxes.tolist()

    imgs = list(getattr(result, "imgs", []) or [])
    if not imgs or line_count == 0:
        return []

    height, width = imgs[0].shape[:2]
    box = [
        [0.0, 0.0],
        [float(width - 1), 0.0],
        [float(width - 1), float(height - 1)],
        [0.0, float(height - 1)],
    ]
    return [box for _ in range(line_count)]


if __name__ == "__main__":
    main()
