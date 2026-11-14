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


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "crates/rapidocr-core/tests/fixtures"
    root.mkdir(parents=True, exist_ok=True)
    write_model(root, "constant-det.onnx", [1, 3, "height", "width"], np.zeros((1, 1, 8, 8)))
    prediction = np.array([[[0, 1, 0], [1, 0, 0]]])  # CTC: "A", then blank.
    write_model(root, "constant-rec.onnx", [1, 3, 48, "width"], prediction)
    write_model(root, "wrong-batch-rec.onnx", [1, 3, 48, "width"], np.repeat(prediction, 2, axis=0))
