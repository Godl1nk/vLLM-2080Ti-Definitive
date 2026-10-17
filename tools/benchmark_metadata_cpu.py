#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure graph metadata CPU staging without importing GPU runtime extensions.

Loads the actual helper definitions from source, using real CPU tensors. This
isolates Python traversal/copy dispatch, not GPU latency or model throughput.
Run before committing with --baseline-ref HEAD to compare working-tree edits.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import statistics
import subprocess
import timeit
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


def load_helpers(root: Path, ref: str | None = None) -> dict[str, Any]:
    """Load dependency-free production helpers without initializing vLLM."""
    namespace = {
        "torch": torch,
        "dataclasses": dataclasses,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "Any": Any,
        "np": np,
    }
    targets = {
        "vllm/compilation/cuda_graph.py": {"_clone_tensor_tree", "_copy_tensor_tree"},
        "vllm/v1/worker/gpu/cudagraph_utils.py": {
            "_copy_fullgraph_metadata",
            "_FULLGRAPH_TOPOLOGY_FIELDS",
        },
        "vllm/v1/attention/backends/utils.py": {
            "split_decodes_and_prefills",
            "reorder_batch_to_split_decodes_and_prefills",
        },
    }
    for relative, names in targets.items():
        source = (
            subprocess.run(
                ["git", "show", f"{ref}:{relative}"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            if ref
            else (root / relative).read_text(encoding="utf-8")
        )
        tree = ast.parse(source, filename=relative)
        selected = []
        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name in names
                or isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in names
                    for target in node.targets
                )
            ):
                selected.append(node)
        exec(
            compile(ast.Module(body=selected, type_ignores=[]), relative, "exec"),
            namespace,
        )
    return namespace


@dataclasses.dataclass
class Metadata:
    num_prefills: int
    num_decodes: int
    num_actual_tokens: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    nested: dict[str, Any]


def make_metadata() -> Metadata:
    return Metadata(
        0,
        1,
        8,
        torch.tensor([4096], dtype=torch.int32),
        torch.arange(256, dtype=torch.int32).reshape(1, -1),
        torch.tensor([0, 8], dtype=torch.int32),
        {
            "positions": torch.arange(8),
            "optional": None,
            "labels": [False, 0, 1.0, "decode"],
        },
    )


def benchmark(
    helpers: dict[str, Any], layers: int, iterations: int, repeats: int
) -> dict[str, float]:
    runtime = make_metadata()
    captured = make_metadata()
    src = {f"layer.{index}": runtime for index in range(layers)}
    dst = {key: captured for key in src}
    compiled_dst = helpers["_clone_tensor_tree"](dst)
    cases = {
        "compiled_shared_metadata": lambda: helpers["_copy_tensor_tree"](
            src, compiled_dst
        ),
        "v2_shared_metadata": lambda: helpers["_copy_fullgraph_metadata"](src, dst),
        "compiled_single_metadata": lambda: helpers["_copy_tensor_tree"](
            runtime, compiled_dst["layer.0"]
        ),
        "v2_single_metadata": lambda: helpers["_copy_fullgraph_metadata"](
            runtime, captured
        ),
    }
    result = {}
    for name, fn in cases.items():
        for _ in range(100):
            fn()
        samples = timeit.repeat(fn, number=iterations, repeat=repeats)
        result[name] = statistics.median(samples) * 1e6 / iterations
    return result


def make_batch(categories: list[int]) -> tuple[Any, Any]:
    """Create CPU batch state with the same swap contract as the runner."""
    # decode, short extend, long extend, first prefill
    states = ((1, 10, 10), (1, 10, 20), (8, 10, 20), (8, 0, 20))
    ids = [str(i) for i in range(len(categories))]
    computed = np.array([states[c][1] for c in categories], dtype=np.int32)
    prompts = np.array([states[c][2] for c in categories], dtype=np.int32)

    def swap_states(i: int, j: int) -> None:
        ids[i], ids[j] = ids[j], ids[i]
        computed[i], computed[j] = computed[j], computed[i]
        prompts[i], prompts[j] = prompts[j], prompts[i]

    return (
        SimpleNamespace(
            req_ids=ids,
            num_computed_tokens_cpu=computed,
            num_prompt_tokens=prompts,
            swap_states=swap_states,
        ),
        SimpleNamespace(
            num_scheduled_tokens={
                str(i): states[c][0] for i, c in enumerate(categories)
            }
        ),
    )


def benchmark_attention(
    helpers: dict[str, Any], iterations: int, repeats: int
) -> dict[str, float]:
    result = {}
    for size in (1, 4, 16, 32):
        batch, output = make_batch([0] * size)
        split = helpers["split_decodes_and_prefills"]
        reorder = helpers["reorder_batch_to_split_decodes_and_prefills"]
        offsets = torch.arange(size + 1, dtype=torch.int32) * 8
        metadata = SimpleNamespace(
            num_reqs=size,
            num_actual_tokens=size * 8,
            max_query_len=8,
            query_start_loc_cpu=offsets,
        )
        cases = {
            "ordered_reorder": partial(reorder, batch, output),
            "speculative_split": partial(split, metadata, 8, True),
            "plain_decode_split": partial(split, metadata, 8, False),
        }
        mixed_offsets = torch.tensor(
            [0, *range(8, size * 8, 8), (size - 1) * 8 + 256],
            dtype=torch.int32,
        )
        mixed = SimpleNamespace(
            num_reqs=size,
            num_actual_tokens=(size - 1) * 8 + 256,
            max_query_len=256,
            query_start_loc_cpu=mixed_offsets,
        )
        cases["mixed_split"] = partial(split, mixed, 8, True)
        if size >= 4:
            # Resetting an unsorted batch is included in both timings, since
            # reordering mutates it. Do not mistake fixture setup for runtime.
            cases["unsorted_reorder_with_setup"] = partial(
                reorder_fresh, reorder, [3, 2, 1, 0] * (size // 4)
            )
        for name, fn in cases.items():
            for _ in range(100):
                fn()
            samples = timeit.repeat(fn, number=iterations, repeat=repeats)
            result[f"{name}_{size}"] = statistics.median(samples) * 1e6 / iterations
    return result


def reorder_fresh(reorder: Any, categories: list[int]) -> bool:
    batch, output = make_batch(categories)
    return reorder(batch, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--suite", choices=("graph", "attention"), default="graph")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.layers, args.iterations, args.repeats) <= 0:
        parser.error("layers, iterations and repeats must be positive")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    report = {
        "scope": "synthetic CPU metadata staging; not GPU/model throughput",
        "torch": torch.__version__,
        "layers": args.layers,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "suite": args.suite,
    }

    def run(helpers: dict[str, Any]) -> dict[str, float]:
        if args.suite == "attention":
            return benchmark_attention(helpers, args.iterations, args.repeats)
        return benchmark(helpers, args.layers, args.iterations, args.repeats)

    if args.baseline_ref:
        report["baseline_ref"] = args.baseline_ref
        report["baseline_us"] = run(load_helpers(root, args.baseline_ref))
    report["current_us"] = run(load_helpers(root))
    if args.baseline_ref:
        report["speedup"] = {
            name: report["baseline_us"][name] / value
            for name, value in report["current_us"].items()
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
