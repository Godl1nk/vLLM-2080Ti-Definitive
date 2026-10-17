# SPDX-License-Identifier: Apache-2.0
"""CPU attention planning checks without importing GPU runtime extensions."""

import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmark_metadata_cpu import load_helpers, make_batch  # noqa: E402

HELPERS = load_helpers(ROOT)
split = HELPERS["split_decodes_and_prefills"]
reorder = HELPERS["reorder_batch_to_split_decodes_and_prefills"]


def metadata(lengths, *, total=None, prefilling=None):
    offsets = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=len(lengths),
        num_actual_tokens=sum(lengths) if total is None else total,
        max_query_len=max(lengths, default=0),
        query_start_loc_cpu=offsets,
        is_prefilling=prefilling,
    )


class AttentionCpuTests(unittest.TestCase):
    def test_single_request_never_reorders(self):
        for category in range(4):
            batch, output = make_batch([category])
            # A single request needs neither classification arrays nor swaps.
            with patch.object(HELPERS["np"], "array", side_effect=AssertionError):
                self.assertFalse(reorder(batch, output))
            self.assertEqual(batch.req_ids, ["0"])

    def test_empty_batch_never_reorders(self):
        batch, output = make_batch([])
        self.assertFalse(reorder(batch, output))

    def test_all_region_combinations_preserve_state(self):
        for size in range(1, 6):
            for categories in itertools.product(range(4), repeat=size):
                batch, output = make_batch(list(categories))
                computed = batch.num_computed_tokens_cpu.copy()
                prompts = batch.num_prompt_tokens.copy()
                changed = reorder(batch, output)
                ordered = [categories[int(key)] for key in batch.req_ids]
                self.assertEqual(ordered, sorted(categories))
                self.assertEqual(changed, list(categories) != sorted(categories))
                for index, key in enumerate(batch.req_ids):
                    self.assertEqual(
                        batch.num_computed_tokens_cpu[index], computed[int(key)]
                    )
                    self.assertEqual(batch.num_prompt_tokens[index], prompts[int(key)])

    def test_single_speculative_decode(self):
        self.assertEqual(split(metadata([8]), 8, True), (1, 0, 8, 0))

    def test_padded_uniform_preserves_actual_token_count(self):
        self.assertEqual(
            split(metadata([8, 8, 0, 0], total=32), 8, True), (4, 0, 32, 0)
        )

    def test_mixed_prefill_keeps_first_boundary(self):
        self.assertEqual(split(metadata([2, 2, 8, 16]), 4, True), (2, 2, 4, 24))
        self.assertEqual(split(metadata([2, 1, 2, 8]), 4, True), (1, 3, 2, 11))
        self.assertEqual(split(metadata([0, 2, 8]), 4, True), (1, 2, 0, 10))

    def test_all_prefill(self):
        self.assertEqual(split(metadata([16]), 8, True), (0, 1, 0, 16))

    def test_nonuniform_decode_and_prefill(self):
        self.assertEqual(split(metadata([1, 2, 8]), 4, False), (2, 1, 3, 8))

    def test_large_batch_fallback_and_cutoff(self):
        for size in (16, 17, 32):
            self.assertEqual(
                split(metadata([8] * size), 8, True), (size, 0, size * 8, 0)
            )
            lengths = [8] * (size - 1) + [16]
            self.assertEqual(
                split(metadata(lengths), 8, True), (size - 1, 1, (size - 1) * 8, 16)
            )

    def test_explicit_prefill_mask_fallback(self):
        m = metadata([1, 1, 8], prefilling=torch.tensor([False, True, True]))
        self.assertEqual(split(m, 8, False, False), (1, 2, 1, 9))


if __name__ == "__main__":
    unittest.main()
