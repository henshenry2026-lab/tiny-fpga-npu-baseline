"""End-to-end tests for the buffered batch-one GEMV model runner."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import npu_mapper as mapper


def tiny_model():
    return {"format": "xilinx-gpu.linear.v1", "input_scale": 1, "layers": [
        {"name": "hidden", "weights": [[1, -1, 3], [2, 1, -2]],
         "weight_scales": [1, 1, 1], "bias": [0, 0, 0], "output_scale": 1, "relu": True},
        {"name": "output", "weights": [[1, 2], [0, 1], [0, -1]],
         "weight_scales": [1, 1], "bias": [0, 0], "output_scale": 1}]
    }


class GemvModelBundleTests(unittest.TestCase):
    def test_export_writes_initial_activation_and_weight_only_model_stream(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = mapper.export_bundle(tiny_model(), [[1, 2]], root, lanes=2, backend="gemv")
            self.assertEqual((root / "inputs.txt").read_text().splitlines(), ["1 2"])
            packed = (root / "model-weights.txt").read_text().splitlines()
            self.assertEqual(packed, ["ff01", "0102", "0003", "00fe", "0201", "0100", "ff00"])
            self.assertEqual(len(packed), sum(layer["schedule"]["accepted_beats"]
                                              for layer in report["layers"]))

    def test_rtl_model_runner_matches_final_layer_under_backpressure(self):
        from hdl_tools import find_tool
        from npu_mapper_sim import simulate_gemv_model_bundle
        try:
            find_tool("iverilog")
            find_tool("vvp")
        except RuntimeError as error:
            self.skipTest(str(error))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = mapper.export_bundle(tiny_model(), [[1, 2]], root, lanes=2, backend="gemv")
            checked = simulate_gemv_model_bundle(root, stalls=True)
            self.assertEqual(checked["rtl_validation"]["model_runner"]["status"], "passed")
            expected = report["layers"][-1]["int8_output"][0]
            results = [int(value) for value in (root / "rtl-model-results.txt").read_text().split()]
            self.assertEqual(results, expected)
            self.assertEqual(checked["rtl_validation"]["model_runner"]["intermediate_outputs_hosted"], 0)

    def test_model_runner_rejects_overlong_initial_activation(self):
        from hdl_tools import find_tool
        from npu_mapper_sim import simulate_gemv_model_bundle
        try:
            find_tool("iverilog")
            find_tool("vvp")
        except RuntimeError as error:
            self.skipTest(str(error))

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mapper.export_bundle(tiny_model(), [[1, 2]], root, lanes=2, backend="gemv")
            with self.assertRaisesRegex(RuntimeError, "input.*capacity"):
                simulate_gemv_model_bundle(root, stalls=False, max_features=1)
            failed = json.loads((root / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["rtl_validation"]["model_runner"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
