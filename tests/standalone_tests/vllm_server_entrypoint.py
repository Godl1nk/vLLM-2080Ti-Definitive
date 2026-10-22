#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_loader(
    "vllm_server_entry",
    importlib.machinery.SourceFileLoader("vllm_server_entry", str(ROOT / "vllm-server")),
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

FP8_QUANTIZATION = {
    "quant_method": "compressed-tensors",
    "config_groups": {
        "group_0": {
            "weights": {"num_bits": 8, "type": "float"},
            "input_activations": {"num_bits": 8, "type": "float"},
        }
    },
}


def write_model(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")


class EntryPointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.vision = base / "Qwen3.8-27B"
        self.text = base / "Qwen3.8-27B-text"
        self.int4 = base / "Qwen3.8-27B-int4"
        write_model(self.vision, {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5VLForConditionalGeneration"],
            "vision_config": {"depth": 27},
            "quantization_config": FP8_QUANTIZATION,
            "max_position_embeddings": 262144,
        })
        write_model(self.text, {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForCausalLM"],
            "quantization_config": FP8_QUANTIZATION,
            "max_position_embeddings": 262144,
        })
        write_model(self.int4, {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForCausalLM"],
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int"}}},
            },
            "max_position_embeddings": 262144,
        })
        self.gpus = [(0, "NVIDIA GeForce RTX 2080 Ti"), (1, "NVIDIA GeForce RTX 2080 Ti")]
        original = module.list_gpus
        module.list_gpus = lambda: self.gpus
        self.addCleanup(setattr, module, "list_gpus", original)

    def select(self, model):
        config = module.model_config(model)
        family = module.guess_family(model, config)
        quantization = module.guess_quantization(model, config)
        multimodal = module.is_multimodal(config)
        profile = module.resolve_profile("", "0,1", self.gpus, family, quantization, multimodal)
        return family, quantization, multimodal, profile

    def argv(self, model, **options):
        family, quantization, multimodal, profile = self.select(model)
        values = {"model": str(model), "host": "127.0.0.1", "port": "5912", "devices": "0,1"}
        values.update(options)
        provided = set()
        passthrough = []
        argv = module.build_argv(dict(values), provided, passthrough, profile, family, quantization, multimodal)
        return argv

    def test_quantization_detection(self):
        self.assertEqual(module.guess_quantization(self.vision, module.model_config(self.vision)), "fp8")
        self.assertEqual(module.guess_quantization(self.int4, module.model_config(self.int4)), "compressed-tensors")

    def test_auto_profile_selection(self):
        _, _, _, profile = self.select(self.vision)
        self.assertEqual(profile.name, "mtp4-fp8kv-1x186K-text-image.env")
        _, _, _, profile = self.select(self.text)
        self.assertEqual(profile.name, "mtp4-fp8kv-1x262K-text-only.env")
        _, _, _, profile = self.select(self.int4)
        self.assertEqual(profile.name, "mtp4-fp8kv-2x229K-text-only.env")

    def test_vision_route_argv(self):
        argv = self.argv(self.vision)
        joined = " ".join(argv)
        self.assertIn("--model " + str(self.vision), joined)
        self.assertIn("--tensor-parallel-size 2", joined)
        self.assertIn("--max-model-len 186368", joined)
        self.assertIn("--kv-cache-dtype fp8", joined)
        self.assertIn("--quantization fp8", joined)
        self.assertIn("--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":4}", joined)
        self.assertIn("--limit-mm-per-prompt {\"image\":64,\"video\":0,\"audio\":0}", joined)
        self.assertIn("--additional-config {\"gdn_prefill_backend\":\"flashqla_legacy\"}", joined)
        self.assertIn("--mamba-cache-mode align", joined)
        self.assertIn("--reasoning-parser qwen3", joined)
        self.assertIn(
            "--compilation-config {\"cudagraph_mode\":\"FULL_AND_PIECEWISE\","
            "\"cudagraph_capture_sizes\":[5],\"max_cudagraph_capture_size\":5}",
            joined,
        )
        self.assertIn("--port 5912", joined)
        self.assertNotIn("--language-model-only", argv)

    def test_text_route_argv(self):
        argv = self.argv(self.text)
        joined = " ".join(argv)
        self.assertIn("--max-model-len 262144", joined)
        self.assertIn("--language-model-only", argv)
        self.assertIn("--skip-mm-profiling", argv)
        self.assertIn("--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":4}", joined)

    def test_int4_route_argv(self):
        argv = self.argv(self.int4)
        joined = " ".join(argv)
        self.assertIn("--quantization compressed-tensors", joined)
        self.assertIn("--max-model-len 229376", joined)
        self.assertIn("--max-num-seqs 2", joined)
        self.assertIn("--prefill-batch-barrier", argv)
        self.assertIn(
            "--compilation-config {\"cudagraph_mode\":\"FULL_AND_PIECEWISE\","
            "\"cudagraph_capture_sizes\":[5,10],\"max_cudagraph_capture_size\":10}",
            joined,
        )

    def test_explicit_flags_win(self):
        argv = self.argv(self.vision, max_model_len="8192", kv_cache_dtype="float16")
        joined = " ".join(argv)
        self.assertIn("--max-model-len 8192", joined)
        self.assertIn("--kv-cache-dtype float16", joined)
        self.assertNotIn("--max-model-len 186368", joined)
        self.assertNotIn("--kv-cache-dtype fp8", joined)

    def test_passthrough_flags_are_forwarded(self):
        family, quantization, multimodal, profile = self.select(self.vision)
        values = {"model": str(self.vision), "host": "127.0.0.1", "port": "5912", "devices": "0,1"}
        argv = module.build_argv(
            values, {"--seed"}, ["--seed", "7"], profile, family, quantization, multimodal
        )
        self.assertEqual(argv[-2:], ["--seed", "7"])

    def test_parsing_accepts_positional_model_and_equals(self):
        options, provided, passthrough, profile, print_config = module.parse_args(
            ["--host=0.0.0.0", "--port=5912", "--max-model-len", "8192", "--print-config", str(self.vision)]
        )
        self.assertEqual(options["host"], "0.0.0.0")
        self.assertEqual(options["port"], "5912")
        self.assertEqual(options["max_model_len"], "8192")
        self.assertEqual(options["model"], str(self.vision))
        self.assertIn("--max-model-len", provided)
        self.assertTrue(print_config)
        self.assertEqual(passthrough, [])

    def test_parse_rejects_missing_model(self):
        with self.assertRaises(SystemExit):
            module.parse_args(["--host", "127.0.0.1"])


if __name__ == "__main__":
    unittest.main()
