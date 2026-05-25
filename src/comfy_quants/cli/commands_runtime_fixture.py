"""Runtime-fixture generation commands."""

from __future__ import annotations

from comfy_quants.algorithms.awq_w4a16.runtime_fixture import (
    DEFAULT_AWQ_RUNTIME_FIXTURE_BATCH,
    DEFAULT_AWQ_RUNTIME_FIXTURE_FILENAME,
    DEFAULT_AWQ_RUNTIME_FIXTURE_K,
    DEFAULT_AWQ_RUNTIME_FIXTURE_LAYER_PREFIX,
    DEFAULT_AWQ_RUNTIME_FIXTURE_N,
    DEFAULT_AWQ_RUNTIME_FIXTURE_REPORT_FILENAME,
    DEFAULT_AWQ_RUNTIME_FIXTURE_SCALE_DTYPE,
    DEFAULT_AWQ_RUNTIME_FIXTURE_SEED,
    AwqW4A16RuntimeFixtureConfig,
    write_awq_w4a16_runtime_fixture,
)
from comfy_quants.algorithms.int4_svdquant.runtime_fixture import (
    DEFAULT_RUNTIME_FIXTURE_BATCH,
    DEFAULT_RUNTIME_FIXTURE_FILENAME,
    DEFAULT_RUNTIME_FIXTURE_K,
    DEFAULT_RUNTIME_FIXTURE_LAYER_PREFIX,
    DEFAULT_RUNTIME_FIXTURE_N,
    DEFAULT_RUNTIME_FIXTURE_RANK,
    DEFAULT_RUNTIME_FIXTURE_REPORT_FILENAME,
    DEFAULT_RUNTIME_FIXTURE_SEED,
    SVDQuantW4A4RuntimeFixtureConfig,
    write_svdquant_w4a4_runtime_fixture,
)
from comfy_quants.backends.runtime_fixture_validation import (
    DEFAULT_RUNTIME_FIXTURE_OUTPUT_VALIDATION_REPORT_FILENAME,
    validate_runtime_fixture_output,
)
from comfy_quants.backends.svdquant_runtime_like_validation import (
    DEFAULT_SVDQUANT_RUNTIME_LIKE_VALIDATION_REPORT_FILENAME,
    validate_svdquant_runtime_like_harness_report,
)
from comfy_quants.backends.int4_runtime_readiness import (
    DEFAULT_INT4_RUNTIME_READINESS_REPORT_FILENAME,
    build_int4_runtime_readiness_report,
)
from comfy_quants.cli.common import ensure_dir, print_json
from comfy_quants.formats.svdquant_w4a4 import LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING, LOWRANK_BRANCH_INPUT_BASIS_RAW
from comfy_quants.utils.jsonio import write_json


def register(subparsers):
    parser = subparsers.add_parser(
        "make-int4-runtime-fixture",
        help="Write a small SVDQuant W4A4 layer fixture for external runtime parity checks",
    )
    parser.add_argument("--out", required=True, help="Output directory for the fixture safetensors file and JSON report")
    parser.add_argument("--fixture-file", default=DEFAULT_RUNTIME_FIXTURE_FILENAME, help="Fixture safetensors filename")
    parser.add_argument("--report-file", default=DEFAULT_RUNTIME_FIXTURE_REPORT_FILENAME, help="JSON report filename")
    parser.add_argument("--seed", type=int, default=DEFAULT_RUNTIME_FIXTURE_SEED)
    parser.add_argument("--n", type=int, default=DEFAULT_RUNTIME_FIXTURE_N, help="Output channel count; must be a multiple of 128")
    parser.add_argument("--k", type=int, default=DEFAULT_RUNTIME_FIXTURE_K, help="Input channel count; must be a multiple of 64")
    parser.add_argument("--rank", type=int, default=DEFAULT_RUNTIME_FIXTURE_RANK)
    parser.add_argument("--batch", type=int, default=DEFAULT_RUNTIME_FIXTURE_BATCH)
    parser.add_argument(
        "--activation-signedness",
        choices=("signed", "unsigned"),
        default="signed",
        help="Activation W4 oracle signedness recorded in the fixture",
    )
    parser.add_argument(
        "--lowrank-branch-input-basis",
        choices=(LOWRANK_BRANCH_INPUT_BASIS_POST_SMOOTHING, LOWRANK_BRANCH_INPUT_BASIS_RAW),
        default=LOWRANK_BRANCH_INPUT_BASIS_RAW,
        help=(
            "Low-rank branch basis stored in proj_down. raw is the default Kitchen/Nunchaku-compatible basis; "
            "post_smoothing stores proj_down for (x / smooth_factor) for internal reference experiments."
        ),
    )
    parser.add_argument("--layer-prefix", default=DEFAULT_RUNTIME_FIXTURE_LAYER_PREFIX)
    parser.add_argument("--no-bias", action="store_true", help="Omit the bias tensor from the layer fixture")
    parser.add_argument("--no-hash", action="store_true", help="Do not compute the fixture file hash")
    parser.add_argument("--json", action="store_true", help="Print the JSON report to stdout")
    parser.set_defaults(func=run)

    awq_parser = subparsers.add_parser(
        "make-awq-runtime-fixture",
        help="Write a small AWQ W4A16 layer fixture for external runtime parity checks",
    )
    awq_parser.add_argument("--out", required=True, help="Output directory for the fixture safetensors file and JSON report")
    awq_parser.add_argument("--fixture-file", default=DEFAULT_AWQ_RUNTIME_FIXTURE_FILENAME, help="Fixture safetensors filename")
    awq_parser.add_argument("--report-file", default=DEFAULT_AWQ_RUNTIME_FIXTURE_REPORT_FILENAME, help="JSON report filename")
    awq_parser.add_argument("--seed", type=int, default=DEFAULT_AWQ_RUNTIME_FIXTURE_SEED)
    awq_parser.add_argument("--n", type=int, default=DEFAULT_AWQ_RUNTIME_FIXTURE_N, help="Output channel count")
    awq_parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_AWQ_RUNTIME_FIXTURE_K,
        help="Input channel count; must be a positive multiple of 64",
    )
    awq_parser.add_argument("--batch", type=int, default=DEFAULT_AWQ_RUNTIME_FIXTURE_BATCH)
    awq_parser.add_argument(
        "--scale-dtype",
        choices=("source", "float16", "bfloat16", "float32"),
        default=DEFAULT_AWQ_RUNTIME_FIXTURE_SCALE_DTYPE,
        help="Dtype used for stored AWQ scale/zero tensors",
    )
    awq_parser.add_argument("--layer-prefix", default=DEFAULT_AWQ_RUNTIME_FIXTURE_LAYER_PREFIX)
    awq_parser.add_argument("--no-bias", action="store_true", help="Omit the bias tensor from the layer fixture")
    awq_parser.add_argument("--no-hash", action="store_true", help="Do not compute the fixture file hash")
    awq_parser.add_argument("--json", action="store_true", help="Print the JSON report to stdout")
    awq_parser.set_defaults(func=run_awq)

    validate_parser = subparsers.add_parser(
        "validate-runtime-fixture-output",
        help="Compare an external runtime output safetensors tensor against a runtime fixture oracle",
    )
    validate_parser.add_argument(
        "--fixture",
        required=True,
        help="Fixture safetensors file produced by make-int4-runtime-fixture or make-awq-runtime-fixture",
    )
    validate_parser.add_argument(
        "--output",
        required=True,
        help="External runtime output safetensors file containing the actual output tensor",
    )
    validate_parser.add_argument(
        "--expected-tensor",
        default="fixture.expected_output",
        help="Tensor name in the fixture file used as the oracle output",
    )
    validate_parser.add_argument(
        "--actual-tensor",
        default="runtime.output",
        help="Tensor name in the external runtime output file",
    )
    validate_parser.add_argument("--atol", type=float, default=1.0e-4, help="Absolute tolerance for torch.allclose")
    validate_parser.add_argument("--rtol", type=float, default=1.0e-4, help="Relative tolerance for torch.allclose")
    validate_parser.add_argument("--out", required=True, help="Output directory for the validation report")
    validate_parser.add_argument(
        "--report-file",
        default=DEFAULT_RUNTIME_FIXTURE_OUTPUT_VALIDATION_REPORT_FILENAME,
        help="JSON report filename",
    )
    validate_parser.add_argument("--json", action="store_true", help="Print the JSON report to stdout")
    validate_parser.set_defaults(func=run_validate_output)

    svd_runtime_like_parser = subparsers.add_parser(
        "validate-svdquant-runtime-like-report",
        help="Validate SVDQuant W4A4 runtime-like parity metrics from an external single-layer harness report",
    )
    svd_runtime_like_parser.add_argument(
        "--harness-report",
        required=True,
        help="JSON report produced by the external SVDQuant W4A4 single-layer harness",
    )
    svd_runtime_like_parser.add_argument("--atol", type=float, default=1.0e-6, help="Maximum allowed absolute error")
    svd_runtime_like_parser.add_argument("--rtol", type=float, default=1.0e-6, help="Maximum allowed relative error")
    svd_runtime_like_parser.add_argument(
        "--expected-dtype",
        default="bfloat16",
        help="Expected runtime dtype recorded in the harness report; use empty string to skip this check",
    )
    svd_runtime_like_parser.add_argument(
        "--allow-non-packed-layout",
        action="store_true",
        help="Do not require a packed assignment layout in the harness report",
    )
    svd_runtime_like_parser.add_argument("--out", required=True, help="Output directory for the validation report")
    svd_runtime_like_parser.add_argument(
        "--report-file",
        default=DEFAULT_SVDQUANT_RUNTIME_LIKE_VALIDATION_REPORT_FILENAME,
        help="JSON report filename",
    )
    svd_runtime_like_parser.add_argument("--json", action="store_true", help="Print the JSON report to stdout")
    svd_runtime_like_parser.set_defaults(func=run_validate_svdquant_runtime_like_report)

    readiness_parser = subparsers.add_parser(
        "validate-int4-runtime-readiness",
        help="Aggregate INT4 runtime parity reports into a publishability gate checklist",
    )
    readiness_parser.add_argument(
        "--svdquant-report",
        help="Report from validate-runtime-fixture-output for a SVDQuant W4A4 fixture",
    )
    readiness_parser.add_argument(
        "--awq-report",
        help="Report from validate-runtime-fixture-output for an AWQ W4A16 fixture",
    )
    readiness_parser.add_argument(
        "--mixed-dispatch-report",
        help="External report proving mixed SVDQuant W4A4 plus AWQ W4A16 dispatch",
    )
    readiness_parser.add_argument(
        "--full-inference-report",
        help="External report proving full Qwen-Image-Edit checkpoint load and PNG inference",
    )
    readiness_parser.add_argument("--out", required=True, help="Output directory for the readiness report")
    readiness_parser.add_argument(
        "--report-file",
        default=DEFAULT_INT4_RUNTIME_READINESS_REPORT_FILENAME,
        help="JSON report filename",
    )
    readiness_parser.add_argument("--json", action="store_true", help="Print the JSON report to stdout")
    readiness_parser.set_defaults(func=run_runtime_readiness)


def run(args) -> int:
    out = ensure_dir(args.out)
    config = SVDQuantW4A4RuntimeFixtureConfig(
        seed=args.seed,
        n=args.n,
        k=args.k,
        rank=args.rank,
        batch=args.batch,
        activation_signedness=args.activation_signedness,
        lowrank_branch_input_basis=args.lowrank_branch_input_basis,
        include_bias=not args.no_bias,
        layer_prefix=args.layer_prefix,
    )
    written = write_svdquant_w4a4_runtime_fixture(
        out,
        config=config,
        fixture_filename=args.fixture_file,
        report_filename=args.report_file,
        hash_fixture=not args.no_hash,
    )
    if args.json:
        print_json(written.report)
    else:
        print(
            "runtime fixture written to "
            f"{written.fixture_path}; report={written.report_path}; status={written.report['status']}"
        )
    return 0 if written.report["status"] == "fixture_written" else 2


def run_awq(args) -> int:
    out = ensure_dir(args.out)
    config = AwqW4A16RuntimeFixtureConfig(
        seed=args.seed,
        n=args.n,
        k=args.k,
        batch=args.batch,
        scale_dtype=args.scale_dtype,
        include_bias=not args.no_bias,
        layer_prefix=args.layer_prefix,
    )
    written = write_awq_w4a16_runtime_fixture(
        out,
        config=config,
        fixture_filename=args.fixture_file,
        report_filename=args.report_file,
        hash_fixture=not args.no_hash,
    )
    if args.json:
        print_json(written.report)
    else:
        print(
            "AWQ runtime fixture written to "
            f"{written.fixture_path}; report={written.report_path}; status={written.report['status']}"
        )
    return 0 if written.report["status"] == "fixture_written" else 2


def run_validate_output(args) -> int:
    out = ensure_dir(args.out)
    report = validate_runtime_fixture_output(
        args.fixture,
        args.output,
        expected_tensor=args.expected_tensor,
        actual_tensor=args.actual_tensor,
        atol=args.atol,
        rtol=args.rtol,
    ).to_dict()
    report_path = out / args.report_file
    write_json(report_path, report)
    if args.json:
        print_json(report)
    else:
        print(f"runtime fixture output validation report written to {report_path}; status={report['status']}")
    return 0 if report["status"] == "passed" else 2


def run_validate_svdquant_runtime_like_report(args) -> int:
    out = ensure_dir(args.out)
    expected_dtype = None if args.expected_dtype == "" else args.expected_dtype
    report = validate_svdquant_runtime_like_harness_report(
        args.harness_report,
        atol=args.atol,
        rtol=args.rtol,
        expected_dtype=expected_dtype,
        require_packed_layout=not args.allow_non_packed_layout,
    ).to_dict()
    report_path = out / args.report_file
    write_json(report_path, report)
    if args.json:
        print_json(report)
    else:
        print(f"SVDQuant runtime-like validation report written to {report_path}; status={report['status']}")
    return 0 if report["status"] == "passed" else 2


def run_runtime_readiness(args) -> int:
    out = ensure_dir(args.out)
    report = build_int4_runtime_readiness_report(
        svdquant_report_path=args.svdquant_report,
        awq_report_path=args.awq_report,
        mixed_dispatch_report_path=args.mixed_dispatch_report,
        full_inference_report_path=args.full_inference_report,
    )
    report_path = out / args.report_file
    write_json(report_path, report)
    if args.json:
        print_json(report)
    else:
        print(f"INT4 runtime readiness report written to {report_path}; status={report['status']}")
    return 0 if report["status"] == "passed" else 2
