"""Diagnostic command."""

from __future__ import annotations

import argparse

from comfy_quants.algorithms.registry import list_algorithms
from comfy_quants.cli.common import print_json
from comfy_quants.comfy.artifact_contracts import get_artifact_contract_index
from comfy_quants.formats.registry import list_formats
from comfy_quants.model_adapters.registry import list_adapters
from comfy_quants.registry.global_registry import registry
from comfy_quants.utils.system_info import collect_system_info


def register(subparsers):
    parser = subparsers.add_parser("doctor", help=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.set_defaults(func=run)


def run(args) -> int:
    info = collect_system_info()
    info["artifact_contracts"] = get_artifact_contract_index().to_dict()
    info["available_adapters"] = list_adapters()
    info["available_algorithms"] = list_algorithms()
    info["available_formats"] = list_formats()
    info["available_backends"] = registry.list_backends()
    if args.json:
        print_json(info)
    else:
        contracts = info["artifact_contracts"]
        print(f"Comfy Quants {info['comfy_quants_version']}")
        print(f"Python: {info['python']}")
        print(f"Platform: {info['platform']}")
        print(f"Adapters: {', '.join(info['available_adapters'])}")
        print(f"Algorithms: {', '.join(info['available_algorithms'])}")
        print(f"Formats: {', '.join(info['available_formats'])}")
        print(f"Backends: {', '.join(info['available_backends'])}")
        print(f"Artifact target: {contracts['artifact_target']}")
        print(f"Contract source: {contracts['contract_source']}")
        print(f"Contract mode: {contracts['contract_mode']}")
        print(f"NVIDIA-SMI: {'found' if info['gpu']['nvidia_smi_found'] else 'not found'}")
        if info['gpu']['nvidia_smi']:
            print(f"GPU: {info['gpu']['nvidia_smi']}")
    return 0
