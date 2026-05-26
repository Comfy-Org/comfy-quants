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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comfy_quants", description="CLI-first offline quantization toolkit")
    parser.add_argument("--version", action="version", version="comfy_quants 0.1.0")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:  # noqa: BLE001 - CLI needs centralized conversion
        return handle_cli_error(exc)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
