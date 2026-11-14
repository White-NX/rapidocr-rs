"""Regenerate tiny, deterministic ONNX models for Rust integration tests.

Requires onnx and numpy only when regenerating the checked-in fixtures. Constant
outputs let the tests exercise native session loading, runtime sharing, and
model contracts without depending on OCR weights or recognition accuracy.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def write_model(root: Path, name: str, shape: list, output: np.ndarray) -> None:
    tensor = numpy_helper.from_array(output.astype(np.float32), name="constant")
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"], value=tensor)],
        name,
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(output.shape))],
    )
    model = helper.make_model(
        graph, producer_name="rapidocr-rs-tests", opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, root / name)


def write_failing_recognition_model(root: Path) -> None:
    """Valid metadata, but reshaping a normal input to three elements fails at run time."""
    shape = numpy_helper.from_array(np.array([1, 1, 3], dtype=np.int64), name="shape")
    graph = helper.make_graph(
        [helper.make_node("Reshape", ["image", "shape"], ["output"])],
        "failing-rec.onnx",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 48, "width"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3])],
        [shape],
    )
    model = helper.make_model(
        graph, producer_name="rapidocr-rs-tests", opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, root / "failing-rec.onnx")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "crates/rapidocr-core/tests/fixtures"
    root.mkdir(parents=True, exist_ok=True)
    write_model(root, "constant-det.onnx", [1, 3, "height", "width"], np.zeros((1, 1, 8, 8)))
    prediction = np.array([[[0, 1, 0], [1, 0, 0]]])  # CTC: "A", then blank.
    write_model(root, "constant-rec.onnx", [1, 3, 48, "width"], prediction)
    write_model(root, "wrong-batch-rec.onnx", [1, 3, 48, "width"], np.repeat(prediction, 2, axis=0))
    detection = np.zeros((1, 1, 64, 64))
    detection[0, 0, 8:16, 28:60] = 0.91  # Wide top line.
    detection[0, 0, 44:52, 4:20] = 0.75  # Narrow bottom line.
    write_model(root, "scored-det.onnx", [1, 3, "height", "width"], detection)
    write_model(root, "constant-cls.onnx", [1, 3, 48, 192], np.array([[0.1, 0.9]]))
    write_model(
        root, "paired-cls.onnx", [2, 3, 48, 192], np.array([[0.92, 0.08], [0.16, 0.84]])
    )
    # Recognition batches sort the narrow bottom crop before the wide top crop.
    recognition = np.array([[[0, 0.96, 0], [1, 0, 0]], [[0, 0.2, 0], [1, 0, 0]]])
    write_model(root, "paired-rec.onnx", [2, 3, 48, "width"], recognition)
    write_failing_recognition_model(root)
