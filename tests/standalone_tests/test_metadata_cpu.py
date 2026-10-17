# SPDX-License-Identifier: Apache-2.0
"""Real-tensor helper checks runnable without CUDA/vLLM native extensions."""

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmark_metadata_cpu import load_helpers  # noqa: E402

HELPERS = load_helpers(ROOT)
copy_tree = HELPERS["_copy_tensor_tree"]
copy_fullgraph = HELPERS["_copy_fullgraph_metadata"]


@dataclass
class Metadata:
    num_actual_tokens: int
    values: torch.Tensor


class MetadataCpuTests(unittest.TestCase):
    def test_shared_metadata_copied_once(self):
        runtime = Metadata(2, torch.tensor([7, 8]))
        captured = {
            "num_actual_tokens": None,
            "values": torch.zeros(2, dtype=torch.int64),
        }
        pointer = captured["values"].data_ptr()
        original = torch.Tensor.copy_
        calls = []

        def counted(dst, src, *args, **kwargs):
            calls.append(dst)
            return original(dst, src, *args, **kwargs)

        with patch.object(torch.Tensor, "copy_", counted):
            copy_tree(
                {str(i): runtime for i in range(48)},
                {str(i): captured for i in range(48)},
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(captured["values"].data_ptr(), pointer)
        torch.testing.assert_close(captured["values"], runtime.values)

    def test_intervening_alias_write_keeps_last_writer(self):
        first = Metadata(1, torch.tensor([1]))
        second = Metadata(1, torch.tensor([2]))
        captured = {"values": torch.tensor([0])}
        copy_tree(
            {"a": first, "b": second, "c": first},
            {"a": captured, "b": captured, "c": captured},
        )
        self.assertEqual(captured["values"].item(), 1)

    def test_distinct_destinations_both_updated(self):
        value = torch.tensor([9])
        left, right = torch.tensor([0]), torch.tensor([0])
        copy_tree({"a": value, "b": value}, {"a": left, "b": right})
        torch.testing.assert_close(left, value)
        torch.testing.assert_close(right, value)

    def test_nested_sequences_and_missing_keys(self):
        value = torch.tensor([3])
        destination = torch.tensor([0])
        untouched = torch.tensor([4])
        copy_tree(
            {"x": [(value, None)]}, {"x": [(destination, None)], "missing": untouched}
        )
        torch.testing.assert_close(destination, value)
        self.assertEqual(untouched.item(), 4)

    def test_identical_tensor_needs_no_copy(self):
        value = torch.tensor([1])
        with patch.object(torch.Tensor, "copy_", side_effect=AssertionError("copy")):
            copy_tree(value, value)
            copy_fullgraph(value, value)

    def test_tensor_views_still_copy(self):
        value = torch.tensor([1, 2])
        copy_tree(value[:1], value[1:])
        torch.testing.assert_close(value, torch.tensor([1, 1]))

    def test_fullgraph_preserves_storage_and_checks_topology(self):
        captured = Metadata(2, torch.tensor([0, 0]))
        pointer = captured.values.data_ptr()
        runtime = Metadata(2, torch.tensor([7, 8]))
        copy_fullgraph(runtime, captured)
        self.assertEqual(captured.values.data_ptr(), pointer)
        torch.testing.assert_close(captured.values, runtime.values)
        with self.assertRaisesRegex(RuntimeError, "num_actual_tokens"):
            copy_fullgraph(Metadata(1, runtime.values), captured)

    def test_fullgraph_rejects_tensor_changes(self):
        for value in (torch.zeros(3), torch.zeros(2, dtype=torch.int64), None):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                copy_fullgraph(value, torch.zeros(2))

    def test_fullgraph_rejects_container_changes(self):
        with self.assertRaisesRegex(RuntimeError, "mapping keys"):
            copy_fullgraph({"new": None}, {"old": None})
        with self.assertRaisesRegex(RuntimeError, "sequence shape"):
            copy_fullgraph([None], [None, None])

    def test_scalar_shortcut_keeps_tensor_presence_check(self):
        with self.assertRaisesRegex(RuntimeError, "tensor presence"):
            copy_fullgraph(torch.tensor([1]), None)

    def test_scalar_subclass_keeps_topology_check(self):
        class Scalar(int):
            num_actual_tokens = 1

        runtime = Scalar(0)
        runtime.num_actual_tokens = 2
        with self.assertRaisesRegex(RuntimeError, "num_actual_tokens"):
            copy_fullgraph(runtime, Scalar(0))


if __name__ == "__main__":
    unittest.main()
