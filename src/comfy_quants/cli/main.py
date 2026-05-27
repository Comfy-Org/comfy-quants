"""Comfy Quants CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

from comfy_quants.cli import (
    commands_calib,
    commands_doctor,
    commands_export,
    commands_export_int4,
    commands_export_model,
    commands_inspect,
    commands_inspect_int4,
    commands_jobs,
    commands_quantize,
    commands_quantize_int4,
    commands_qwen_image_edit_int4,
    commands_runtime_fixture,
    commands_validate,
)
from comfy_quants.cli.common import handle_cli_error


HIDDEN_HELP_COMMANDS = {
    "doctor",
    "make-int4-runtime-fixture",
    "make-awq-runtime-fixture",
    "validate-runtime-fixture-output",
    "validate-svdquant-runtime-like-report",
    "validate-int4-runtime-readiness",
}

COMMAND_ALIASES = {
    "quantize-qwen-image-edit-2511-int4": "qwen-image-edit-2511-int4",
}


def _normalize_argv(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        return None
    normalized = list(argv)
    if normalized and normalized[0] in COMMAND_ALIASES:
        normalized[0] = COMMAND_ALIASES[normalized[0]]
    return normalized


def _hide_subcommands_from_help(subparsers: argparse._SubParsersAction, hidden: set[str]) -> None:
    """Keep compatibility commands callable without listing them in public help."""

    subparsers._choices_actions = [  # noqa: SLF001 - argparse has no public API for this
        action
        for action in subparsers._choices_actions  # noqa: SLF001
        if action.dest not in hidden
    ]
    visible = [name for name in subparsers.choices if name not in hidden]
    subparsers.metavar = "{" + ",".join(visible) + "}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comfy-quants", description="Offline quantization toolkit for ComfyUI-loadable checkpoints")
    parser.add_argument("--version", action="version", version="comfy-quants 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands_doctor.register(subparsers)
    commands_inspect.register(subparsers)
    commands_inspect_int4.register(subparsers)
    commands_calib.register(subparsers)
    commands_quantize.register(subparsers)
    commands_quantize_int4.register(subparsers)
    commands_qwen_image_edit_int4.register(subparsers)
    commands_runtime_fixture.register(subparsers)
    commands_validate.register(subparsers)
    commands_export.register(subparsers)
    commands_export_int4.register(subparsers)
    commands_export_model.register(subparsers)
    commands_jobs.register_jobs(subparsers)
    commands_jobs.register_resume(subparsers)
    _hide_subcommands_from_help(subparsers, HIDDEN_HELP_COMMANDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(_normalize_argv(argv))
    try:
        return int(args.func(args) or 0)
    except Exception as exc:  # noqa: BLE001 - CLI needs centralized conversion
        return handle_cli_error(exc)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
