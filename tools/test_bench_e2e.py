from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import bench_e2e


class ProcessSamplingTests(unittest.TestCase):
    @unittest.skipIf(bench_e2e.psutil is None, "psutil is optional")
    def test_samples_transient_memory_and_drains_both_pipes(self):
        # Allocate between calls and free before the child reports its RSS. The
        # old Python benchmark observed only the final, much lower reading.
        script = """
import gc, json, sys, time, psutil
sys.stdout.write('x' * 131072 + '\\n')
sys.stderr.write('y' * 131072)
allocation = bytearray(64 * 1024 * 1024)
for index in range(0, len(allocation), 4096):
    allocation[index] = 1
time.sleep(0.15)
del allocation
gc.collect()
time.sleep(0.05)
print(json.dumps({'rss': psutil.Process().memory_info().rss}))
"""
        stdout, stderr, memory = bench_e2e.run_sampled(
            [sys.executable, "-c", script], Path(__file__).resolve().parents[1], 0.002
        )
        after_call_rss = json.loads(stdout.splitlines()[-1])["rss"]
        self.assertEqual(stderr, "y" * 131072)
        self.assertEqual(stdout.splitlines()[0], "x" * 131072)
        self.assertGreater(memory.peak_rss_bytes, after_call_rss + 32 * 1024 * 1024)

    def test_benchmark_still_runs_without_optional_psutil(self):
        with patch.object(bench_e2e, "psutil", None):
            stdout, stderr, memory = bench_e2e.run_sampled(
                [sys.executable, "-c", "print('done')"], Path(__file__).resolve().parents[1]
            )
        self.assertEqual(stdout.strip(), "done")
        self.assertEqual(stderr, "")
        self.assertIsNone(memory.peak_rss_bytes)


if __name__ == "__main__":
    unittest.main()
