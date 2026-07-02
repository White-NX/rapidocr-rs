from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

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
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        default=None,
        help="Image path relative to the Python RapidOCR repo, or an absolute path.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("fixtures/db_postprocess"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rs_root = args.rs_root.resolve()
    python_repo = resolve_python_repo(args.python_repo)
    python_dir = python_repo / "python"
    sys.path.insert(0, str(python_dir))

    from rapidocr import RapidOCR
    from rapidocr.utils.process_img import apply_vertical_padding

    model_dir = resolve_under(rs_root, args.model_dir)
    base_out_dir = resolve_under(rs_root, args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    images = args.image or [
        Path("python/tests/test_files/ch_en_num.jpg"),
        Path("python/tests/test_files/text_det.jpg"),
        Path("python/tests/test_files/en.jpg"),
        Path("python/tests/test_files/test_letterbox_like.jpg"),
        Path("python/tests/test_files/test_without_det.jpg"),
        Path("python/tests/test_files/text_vertical_words.png"),
        Path("python/tests/test_files/empty_black.jpg"),
        Path("python/tests/test_files/issue_170.png"),
        Path("python/tests/test_files/short.png"),
        Path("python/tests/test_files/return_word_debug.jpg"),
        Path("python/tests/test_files/black_font_color_transparent.png"),
        Path("python/tests/test_files/ch_doc_server.png"),
        Path("python/tests/test_files/check_return_word_len.jpeg"),
        Path("python/tests/test_files/arabic.png"),
        Path("python/tests/test_files/devanagari.jpg"),
        Path("python/tests/test_files/japan.jpg"),
    ]

    engine = RapidOCR(
        params={
            "Global.use_cls": False,
            "Global.model_root_dir": str(model_dir),
        }
    )

    for image in images:
        export_one(
            rs_root,
            python_repo,
            image,
            args.model_dir,
            base_out_dir,
            engine,
            apply_vertical_padding,
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


def export_one(
    rs_root,
    python_repo,
    image,
    model_dir_arg,
    base_out_dir,
    engine,
    apply_vertical_padding,
):
    image_path = fixture_image_path(python_repo, image)
    out_dir = base_out_dir / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img = engine.load_img(image_path)
    img, op_record = engine.preprocess_img(img)
    img, _ = apply_vertical_padding(
        img,
        op_record,
        engine.width_height_ratio,
        engine.min_height,
    )

    detector = engine.text_det
    ori_shape = img.shape[:2]
    detector.preprocess_op = detector.get_preprocess(max(img.shape[0], img.shape[1]))
    input_tensor = detector.preprocess_op(img)
    pred = detector.session(input_tensor)
    boxes, scores = detector.postprocess_op(pred, ori_shape)
    pairs = sorted(
        zip(boxes, scores),
        key=lambda item: (round(float(item[0][0][1]) / 10.0), float(item[0][0][0])),
    )
    if pairs:
        boxes, scores = zip(*pairs)
        boxes = np.asarray(boxes)
        scores = list(scores)
    else:
        boxes = np.empty((0, 4, 2), dtype=np.float32)
        scores = []

    np.save(out_dir / "pred.npy", pred.astype(np.float32))
    metadata = {
        "image": display_path(python_repo, image_path),
        "model_dir": display_path(rs_root, resolve_under(rs_root, model_dir_arg)),
        "dest_shape": [int(ori_shape[1]), int(ori_shape[0])],
        "pred_shape": list(map(int, pred.shape)),
        "boxes": boxes.astype(float).tolist(),
        "scores": [float(score) for score in scores],
    }
    (out_dir / "expected.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
