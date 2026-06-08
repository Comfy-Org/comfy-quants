import os
import subprocess
import sys
import unittest
from typing import Any

from comfy_quants.model_adapters.base import ModelSource
from comfy_quants.model_adapters.qwen_contracts.qwen_image_layered import (
    get_qwen_image_layered_static_contract,
)
from comfy_quants.model_adapters.registry import get_adapter

# Contract values below are verified against the real Qwen/Qwen-Image-Layered
# diffusers checkpoint (transformer/config.json + safetensors index): num_layers 60,
# num_attention_heads 24, attention_head_dim 128 (hidden 3072), joint_attention_dim
# 3584, in/out channels 64/16, use_additional_t_cond=true, and a block whose only
# norms are attn.norm_q/norm_k/norm_added_q/norm_added_k (no img_norm*/txt_norm*),
# with norm_out.linear (no norm_out.norm).
_SOURCE = ModelSource(family="qwen_image_layered", model_id="Qwen/Qwen-Image-Layered")


class TestQwenImageLayeredContract(unittest.TestCase):
    def test_contract_dimensions_and_format(self):
        contract = get_qwen_image_layered_static_contract()
        self.assertEqual(contract.family, "qwen_image_layered")
        self.assertEqual(contract.artifact_target, "comfyui")
        self.assertEqual(contract.contract_mode, "static_adapter_contract")
        self.assertEqual(contract.preferred_format, "fp8_e4m3")
        t = contract.transformer
        self.assertEqual(t.block_prefix, "transformer_blocks")
        self.assertEqual(t.block_count, 60)
        self.assertEqual(t.hidden_size, 3072)
        self.assertEqual(t.intermediate_size, 12288)
        self.assertEqual(t.num_attention_heads, 24)
        self.assertEqual(t.attention_head_dim, 128)
        self.assertEqual(t.joint_attention_dim, 3584)
        self.assertEqual(t.in_channels, 64)
        self.assertEqual(t.out_channels, 16)

    def test_graph_tensor_names_match_real_checkpoint(self):
        adapter = get_adapter("qwen_image_layered")
        _inspection, graph = adapter.inspect(_SOURCE)
        names = {module.name for module in graph.modules}
        # Present in the real checkpoint.
        for present in (
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.add_q_proj",
            "transformer_blocks.0.attn.to_add_out",
            "transformer_blocks.0.attn.norm_added_q",
            "transformer_blocks.0.img_mlp.net.0.proj",
            "transformer_blocks.0.txt_mlp.net.2",
            "transformer_blocks.0.img_mod.1",
            "transformer_blocks.0.txt_mod.1",
            "time_text_embed.addition_t_embedding",
            "norm_out.linear",
            "proj_out",
            "vae",
            "text_encoders.qwen25_7b",
        ):
            self.assertIn(present, names)
        # Absent in the real Qwen-Image-Layered block (unlike base Qwen-Image):
        # no per-block img/txt norms and no norm_out.norm.
        for absent in (
            "transformer_blocks.0.img_norm1",
            "transformer_blocks.0.img_norm2",
            "transformer_blocks.0.txt_norm1",
            "transformer_blocks.0.txt_norm2",
            "norm_out.norm",
        ):
            self.assertNotIn(absent, names)

    def test_additional_t_cond_embedding_is_kept_high_precision(self):
        adapter = get_adapter("qwen_image_layered")
        _inspection, graph = adapter.inspect(_SOURCE)
        by_name = {module.name: module for module in graph.modules}
        emb = by_name["time_text_embed.addition_t_embedding"]
        self.assertEqual(emb.module_type, "Embedding")
        self.assertFalse(emb.quantizable)
        self.assertEqual(emb.default_action, "keep_bf16")
        self.assertEqual(emb.tensors[0].name, "time_text_embed.addition_t_embedding.weight")
        self.assertEqual(emb.tensors[0].shape, [2, 3072])

    def test_graph_metadata_layered_specific(self):
        adapter = get_adapter("qwen_image_layered")
        inspection, graph = adapter.inspect(_SOURCE)
        self.assertEqual(graph.metadata["graph_kind"], "static_model_contract")
        self.assertEqual(graph.metadata["tensor_coverage"], "declared_tensors")
        self.assertEqual(graph.metadata["architecture"], "qwen_image_layered_transformer_2d")
        self.assertEqual(graph.metadata["latent_format"], "wan21")
        self.assertEqual(graph.metadata["model_contract"]["block_count"], 60)
        # Artifact contract is populated (parity with qwen_image / qwen_image_edit).
        self.assertIn("artifact_contract", inspection.metadata)
        self.assertEqual(
            inspection.metadata["artifact_contract"]["schema_version"],
            "qwen_image_layered_contract.v1",
        )
        self.assertEqual(inspection.metadata["artifact_contract"]["artifact_target"], "comfyui")

    def test_quantizable_module_count(self):
        adapter = get_adapter("qwen_image_layered")
        inspection, _graph = adapter.inspect(_SOURCE)
        # 14 quantizable linears per block (6 mlp/mod + 8 attn projections) x 60 blocks.
        self.assertEqual(inspection.quantizable_modules, 60 * 14)
        self.assertGreater(inspection.total_parameters, 0)

    def test_graph_metadata_has_production_terms(self):
        adapter = get_adapter("qwen_image_layered")
        _inspection, graph = adapter.inspect(_SOURCE)
        banned = ("place" + "holder", "sym" + "bolic", "not_" + "implemented", "up" + "stream")

        def walk(value: Any):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield key
                    yield from walk(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from walk(item)
            else:
                yield value

        for value in walk(graph.metadata):
            if isinstance(value, str):
                self.assertFalse(any(term in value.lower() for term in banned), value)

    def test_registered_on_bare_package_import(self):
        # Root comfy_quants/__init__.py must import the adapter for its registration
        # side effect, so a bare `import comfy_quants` lists it (not only via the
        # model_adapters.registry facade). Checked in a clean interpreter.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        src = os.path.join(repo_root, "src")
        env = dict(os.environ)
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        code = "import comfy_quants; print('qwen_image_layered' in comfy_quants.list_model_adapters())"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(result.stdout.strip(), "True", result.stderr)


if __name__ == "__main__":
    unittest.main()
