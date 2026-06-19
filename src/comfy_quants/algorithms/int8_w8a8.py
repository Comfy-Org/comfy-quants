"""INT8 W8A8 quantization planner.

Like ``fp8_static``, the planner only assigns per-module quantize/keep actions and
stamps the target dtype; the int8/ConvRot math lives in the backend writer
(``backends/int8_w8a8_model_export.py``).
"""

from __future__ import annotations

from comfy_quants.algorithms.base import AlgorithmPlanStep
from comfy_quants.algorithms.tensor_index import module_selected_by_policy
from comfy_quants.core.graph import ModelGraph
from comfy_quants.core.policy import QuantPolicy


class Int8W8A8Algorithm:
    name = "int8_w8a8"
    version = "0.1.0"

    def plan(self, graph: ModelGraph, policy: QuantPolicy) -> list[AlgorithmPlanStep]:
        steps: list[AlgorithmPlanStep] = []
        for index, module in enumerate(graph.modules):
            action = "quantize" if module_selected_by_policy(module, policy) else "keep_bf16"
            if not module.quantizable:
                action = module.default_action
            steps.append(AlgorithmPlanStep(
                step_id=f"{index:06d}",
                module_name=module.name,
                action=action,
                algorithm=self.name if action == "quantize" else "none",
                target_dtype=policy.target_dtype if action == "quantize" else "bf16",
            ))
        return steps


from comfy_quants.registry.global_registry import registry  # noqa: E402

registry.register_algorithm(Int8W8A8Algorithm())
