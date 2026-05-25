#!/usr/bin/env bash
# Serial DeepCompressor Qwen-Image-Edit-2511 INT4 -> ComfyUI kitchen tile-pack exporter.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COMFY_QUANTS_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

DEEP_ROOT="${DEEP_ROOT:-/workspace/external/deepcompressor-yidhar}"
NUNCHAKU_ROOT="${NUNCHAKU_ROOT:-/workspace/external/nunchaku}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen-Image-Edit-2511}"
RUNS_ROOT=""
EXPORT_ROOT=""
EXPORT_NAME=""
CANDIDATE="quality-r64"
GPUS="0"
PYTHON_BIN="${PYTHON_BIN:-python}"
MICROMAMBA_ENV="${MICROMAMBA_ENV:-}"
SEARCH_CALIB_PATH="${QWEN_IMAGE_EDIT_2511_SEARCH_CALIB_PATH:-}"

ROUTE="nunchaku-bridge"
RUN_PTQ=false
QUANT_PATH=""
RUN_DIR=""
BASE_COMFY=""
OUTPUT=""
RAW_NUNCHAKU=""
AWQ_GROUP_SIZE="64"
NO_AWQ_MODULATION=false
REUSE=false
DRY_RUN=false
VALIDATE_OUTPUT=true
HASH_OUTPUT=false
NUNCHAKU_MODEL_CLASS="NunchakuQwenImageTransformer2DModel"

usage() {
  cat <<'EOF'
Usage:
  scripts/external/qwen_image_edit_2511_int4_tilepack_pipeline.sh \
    --run-ptq \
    --candidate quality-r64 \
    --gpus 0 \
    --base-comfy /path/to/qwen_edit_2511_kitchen_native_scaffold.safetensors \
    --output /path/to/qwen_edit_2511_int4_comfy_tilepack.safetensors

  scripts/external/qwen_image_edit_2511_int4_tilepack_pipeline.sh \
    --quant-path /path/to/deepcompressor/run/model \
    --base-comfy /path/to/qwen_edit_2511_kitchen_native_scaffold.safetensors \
    --output /path/to/qwen_edit_2511_int4_comfy_tilepack.safetensors

Purpose:
  One serial entrypoint for Qwen-Image-Edit-2511 INT4 export:

    DeepCompressor PTQ artifacts
      -> DeepCompressor Nunchaku split checkpoint
      -> raw Nunchaku single safetensors
      -> ComfyUI/comfy-kitchen kitchen tile-packed safetensors

  Default route is "nunchaku-bridge", which preserves:
    - SVDQuant W4A4 attention/MLP as kitchen_tile_packed_w4a4
    - img_mod.1 / txt_mod.1 as AWQ W4A16, if the required helper branch exists
    - non-quantized tensors from --base-comfy scaffold

Required inputs:
  Either:
    --run-ptq                         launch DeepCompressor search PTQ first
  or:
    --quant-path DIR                  existing PTQ artifact dir containing model.pt/scale.pt/smooth.pt/branch.pt

  For --route nunchaku-bridge:
    --base-comfy FILE                 ComfyUI/kitchen-native scaffold safetensors
                                      (non-quant tensors + target topology)

Common options:
  --deepcompressor-root DIR           DeepCompressor-yidhar repo (default: /workspace/external/deepcompressor-yidhar)
  --nunchaku-root DIR                 Nunchaku checkout with tools/kitchen_native (default: /workspace/external/nunchaku)
  --model-id ID_OR_PATH               HF id or local Qwen-Image-Edit-2511 path (default: Qwen/Qwen-Image-Edit-2511)
  --candidate NAME                    DeepCompressor search candidate for --run-ptq (default: quality-r64)
  --gpus CSV                          CUDA_VISIBLE_DEVICES passed to search launcher (default: 0)
  --search-calib-path DIR             Override QWEN_IMAGE_EDIT_2511_SEARCH_CALIB_PATH for PTQ
  --runs-root DIR                     DeepCompressor runs root (default: <deepcompressor-root>/runs)
  --export-root DIR                   Intermediate/output root (default: <repo>/runs/qwen-image-edit-2511-int4-tilepack-bridge)
  --export-name NAME                  Intermediate split checkpoint directory name
  --raw-nunchaku FILE                 Raw merged Nunchaku safetensors path
  --output FILE                       Final ComfyUI tile-pack safetensors path
  --awq-group-size N                  AWQ W4A16 modulation group size (default: 64)
  --no-awq-modulation                 Pass through legacy converter mode; not recommended for final mixed INT4
  --reuse                             Reuse existing split/raw/final artifacts when present
  --hash-output                       Print SHA256 of final output after validation
  --no-validate                       Do not run final safetensors metadata/key validation
  --dry-run                           Print commands without executing them

Fallback route:
  --route deepcompressor-import       Use this repo's dependency-free DeepCompressor .pt -> SVDQuant tile-pack
                                      importer. This is useful for debugging SVDQuant attention/MLP export only;
                                      it is not the full AWQ-modulation bridge.

Python environment:
  --python-bin BIN                    Python executable (default: python)
  --micromamba-env DIR                Optional micromamba env prefix; command becomes:
                                      micromamba run -p DIR <python-bin>

Examples:
  # Full route, starting from existing DeepCompressor PTQ artifacts:
  scripts/external/qwen_image_edit_2511_int4_tilepack_pipeline.sh \
    --quant-path /workspace/external/deepcompressor-yidhar/runs/.../run-.../model \
    --base-comfy /models/qwen_edit_2511_base_kitchen_native.safetensors \
    --output /models/qwen_edit_2511_quality_r64_int4_tilepack.safetensors \
    --nunchaku-root /path/to/nunchaku-with-tools-kitchen-native \
    --reuse --hash-output

  # Full route, including PTQ launch:
  scripts/external/qwen_image_edit_2511_int4_tilepack_pipeline.sh \
    --run-ptq --candidate quality-r96 --gpus 0 \
    --model-id /models/Qwen-Image-Edit-2511 \
    --search-calib-path /datasets/torch.bfloat16/qwen-image-edit-2511/fmeuler50-g4.0/qdiff/s128 \
    --base-comfy /models/qwen_edit_2511_base_kitchen_native.safetensors \
    --output /models/qwen_edit_2511_quality_r96_int4_tilepack.safetensors
EOF
}

log() { printf '[qwen-int4-tilepack] %s\n' "$*" >&2; }
warn() { printf '[qwen-int4-tilepack][WARN] %s\n' "$*" >&2; }
die() { printf '[qwen-int4-tilepack][ERROR] %s\n' "$*" >&2; exit 1; }

quote_cmd() {
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
}

run_in_dir() {
  local dir="$1"; shift
  printf '+ (cd %q &&' "$dir" >&2
  quote_cmd "$@" >&2
  printf ' )\n' >&2
  if [[ "${DRY_RUN}" == true ]]; then
    return 0
  fi
  (cd "$dir" && "$@")
}

run_here() {
  printf '+' >&2
  quote_cmd "$@" >&2
  printf '\n' >&2
  if [[ "${DRY_RUN}" == true ]]; then
    return 0
  fi
  "$@"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing file: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "missing directory: $path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --deepcompressor-root) DEEP_ROOT="$2"; shift 2 ;;
    --nunchaku-root) NUNCHAKU_ROOT="$2"; shift 2 ;;
    --model-id|--model-path) MODEL_ID="$2"; shift 2 ;;
    --runs-root) RUNS_ROOT="$2"; shift 2 ;;
    --export-root) EXPORT_ROOT="$2"; shift 2 ;;
    --export-name) EXPORT_NAME="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --micromamba-env) MICROMAMBA_ENV="$2"; shift 2 ;;
    --search-calib-path) SEARCH_CALIB_PATH="$2"; shift 2 ;;
    --route) ROUTE="$2"; shift 2 ;;
    --run-ptq) RUN_PTQ=true; shift ;;
    --quant-path) QUANT_PATH="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --base-comfy) BASE_COMFY="$2"; shift 2 ;;
    -o|--output) OUTPUT="$2"; shift 2 ;;
    --raw-nunchaku) RAW_NUNCHAKU="$2"; shift 2 ;;
    --awq-group-size) AWQ_GROUP_SIZE="$2"; shift 2 ;;
    --no-awq-modulation) NO_AWQ_MODULATION=true; shift ;;
    --reuse) REUSE=true; shift ;;
    --hash-output) HASH_OUTPUT=true; shift ;;
    --no-validate) VALIDATE_OUTPUT=false; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) die "unknown argument: $1 (use --help)" ;;
  esac
done

case "$ROUTE" in
  nunchaku-bridge|deepcompressor-import) ;;
  *) die "unsupported --route '$ROUTE'; expected nunchaku-bridge or deepcompressor-import" ;;
esac

if [[ -n "$RUN_DIR" && -z "$QUANT_PATH" ]]; then
  QUANT_PATH="${RUN_DIR%/}/model"
fi
if [[ -z "$RUNS_ROOT" ]]; then
  RUNS_ROOT="${DEEP_ROOT%/}/runs"
fi
if [[ -z "$EXPORT_ROOT" ]]; then
  EXPORT_ROOT="${COMFY_QUANTS_ROOT}/runs/qwen-image-edit-2511-int4-tilepack-bridge"
fi
if [[ -z "$EXPORT_NAME" ]]; then
  EXPORT_NAME="qwen-image-edit-2511-${CANDIDATE}-gptq"
fi
if [[ -z "$RAW_NUNCHAKU" ]]; then
  RAW_NUNCHAKU="${EXPORT_ROOT%/}/${EXPORT_NAME}-raw-nunchaku-int4.safetensors"
fi
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${EXPORT_ROOT%/}/${EXPORT_NAME}-comfy-kitchen-tilepack.safetensors"
fi

PYTHON_CMD=("$PYTHON_BIN")
if [[ -n "$MICROMAMBA_ENV" ]]; then
  PYTHON_CMD=(micromamba run -p "$MICROMAMBA_ENV" "$PYTHON_BIN")
fi

preflight_common() {
  require_dir "$DEEP_ROOT"
  require_file "$DEEP_ROOT/deepcompressor/app/diffusion/ptq.py"
  require_file "$DEEP_ROOT/deepcompressor/backend/nunchaku/convert.py"
  require_file "$DEEP_ROOT/examples/diffusion/scripts/qwen-image-edit-2511-search.py"
  require_file "$DEEP_ROOT/examples/diffusion/scripts/convert_kitchen_native.py"
  if [[ "$ROUTE" == "deepcompressor-import" ]]; then
    require_dir "$COMFY_QUANTS_ROOT/src/comfy_quants"
  fi
}

preflight_nunchaku() {
  require_dir "$NUNCHAKU_ROOT"
  require_file "$NUNCHAKU_ROOT/nunchaku/merge_safetensors.py"
  if [[ ! -d "$NUNCHAKU_ROOT/tools/kitchen_native" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
      warn "Nunchaku checkout lacks tools/kitchen_native; dry-run continues, but real conversion will fail until --nunchaku-root points to the helper branch/checkout."
    else
      die "Nunchaku checkout lacks tools/kitchen_native. convert_kitchen_native.py requires tools/kitchen_native/{interop.py,awq_modulation.py}. Point --nunchaku-root to the helper branch/checkout."
    fi
  fi
  if [[ -z "$BASE_COMFY" ]]; then
    die "--base-comfy is required for --route nunchaku-bridge"
  fi
  if [[ "$DRY_RUN" != true ]]; then
    require_file "$BASE_COMFY"
  fi
}

validate_quant_path() {
  local qp="$1"
  require_dir "$qp"
  require_file "$qp/model.pt"
  require_file "$qp/scale.pt"
  require_file "$qp/smooth.pt"
  require_file "$qp/branch.pt"
  if [[ ! -f "$qp/wgts.pt" ]]; then
    warn "wgts.pt is missing under $qp; conversion does not read it directly, but a normal DeepCompressor export usually has it."
  fi
}

find_latest_model_run() {
  local job_name="$1"
  local latest=""
  latest=$(
    find "$RUNS_ROOT" -type f \( \
      -path "*/${job_name}/run-*/model/model.pt" -o \
      -path "*/${job_name}.RUNNING/run-*/model/model.pt" \
    \) 2>/dev/null \
      | while IFS= read -r model_file; do
          run_dir=$(dirname "$(dirname "$model_file")")
          case "$run_dir" in
            *.ERROR|*.ERROR/*) continue ;;
          esac
          printf '%s %s\n' "$(stat -c '%Y' "$model_file")" "$run_dir"
        done \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )
  [[ -n "$latest" ]] || return 1
  printf '%s\n' "$latest"
}

run_ptq_if_needed() {
  if [[ -n "$QUANT_PATH" ]]; then
    if [[ "$RUN_PTQ" == true ]]; then
      warn "both --quant-path and --run-ptq were given; using existing --quant-path and skipping PTQ."
    fi
    return 0
  fi
  [[ "$RUN_PTQ" == true ]] || die "provide --quant-path/--run-dir, or pass --run-ptq to launch DeepCompressor PTQ"

  local env_args=(
    "PYTHONPATH=${DEEP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    "QWEN_IMAGE_EDIT_2511_MODEL_PATH=${MODEL_ID}"
  )
  if [[ -n "$SEARCH_CALIB_PATH" ]]; then
    env_args+=("QWEN_IMAGE_EDIT_2511_SEARCH_CALIB_PATH=${SEARCH_CALIB_PATH}")
  fi

  log "launching DeepCompressor PTQ search candidate=${CANDIDATE}, gpus=${GPUS}"
  local launcher_python_args=(--python-bin "$PYTHON_BIN")
  if [[ -n "$MICROMAMBA_ENV" ]]; then
    launcher_python_args+=(--micromamba-env "$MICROMAMBA_ENV")
  fi
  run_in_dir "$DEEP_ROOT" env "${env_args[@]}" "${PYTHON_CMD[@]}" \
    examples/diffusion/scripts/qwen-image-edit-2511-search.py \
    launch --gpus "$GPUS" --candidates "$CANDIDATE" "${launcher_python_args[@]}" --wait

  if [[ "$DRY_RUN" == true ]]; then
    QUANT_PATH="${RUNS_ROOT}/<qwen-image-edit-2511-search-${CANDIDATE}>/run-<timestamp>/model"
    return 0
  fi

  local job_name="qwen-image-edit-2511-search-${CANDIDATE}"
  local latest_run
  latest_run=$(find_latest_model_run "$job_name") || die "PTQ finished but no model.pt was found for job $job_name under $RUNS_ROOT"
  RUN_DIR="$latest_run"
  QUANT_PATH="${RUN_DIR%/}/model"
  log "using PTQ artifact dir: $QUANT_PATH"
}

run_deepcompressor_import_route() {
  mkdir -p "$(dirname "$OUTPUT")"
  local env_args=("PYTHONPATH=${COMFY_QUANTS_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}")
  local hash_args=()
  [[ "$HASH_OUTPUT" == true ]] && hash_args+=(--hash-output)
  log "running fallback DeepCompressor import route (SVDQuant attention/MLP only; not full AWQ modulation bridge)"
  run_in_dir "$COMFY_QUANTS_ROOT" env "${env_args[@]}" "${PYTHON_CMD[@]}" \
    -m comfy_quants.cli.main export-int4 \
    --format svdquant_w4a4 \
    --source-format deepcompressor-qwen-image-edit \
    --source "$QUANT_PATH" \
    --out "$OUTPUT" \
    --device auto \
    "${hash_args[@]}" \
    --json
}

run_nunchaku_bridge_route() {
  mkdir -p "$EXPORT_ROOT" "$(dirname "$OUTPUT")" "$(dirname "$RAW_NUNCHAKU")"
  local split_dir="${EXPORT_ROOT%/}/${EXPORT_NAME}"

  if [[ "$REUSE" == true && -f "$split_dir/transformer_blocks.safetensors" && -f "$split_dir/unquantized_layers.safetensors" ]]; then
    log "reuse split Nunchaku checkpoint: $split_dir"
  else
    log "converting DeepCompressor artifacts -> Nunchaku split checkpoint: $split_dir"
    run_in_dir "$DEEP_ROOT" env "PYTHONPATH=${DEEP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_CMD[@]}" \
      -m deepcompressor.backend.nunchaku.convert \
      --quant-path "$QUANT_PATH" \
      --output-root "$EXPORT_ROOT" \
      --model-name "$EXPORT_NAME" \
      --model-path "$MODEL_ID"
  fi

  if [[ "$REUSE" == true && -f "$RAW_NUNCHAKU" ]]; then
    log "reuse raw Nunchaku safetensors: $RAW_NUNCHAKU"
  else
    log "merging Nunchaku split checkpoint -> raw safetensors: $RAW_NUNCHAKU"
    run_in_dir "$DEEP_ROOT" env "PYTHONPATH=${NUNCHAKU_ROOT}:${DEEP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_CMD[@]}" \
      -m nunchaku.merge_safetensors \
      -i "$split_dir" \
      -m "$NUNCHAKU_MODEL_CLASS" \
      -o "$RAW_NUNCHAKU"
  fi

  if [[ "$REUSE" == true && -f "$OUTPUT" ]]; then
    log "reuse final ComfyUI tile-pack safetensors: $OUTPUT"
  else
    local awq_args=(--awq-group-size "$AWQ_GROUP_SIZE")
    [[ "$NO_AWQ_MODULATION" == true ]] && awq_args+=(--no-awq-modulation)
    log "converting raw Nunchaku -> ComfyUI kitchen tile-pack: $OUTPUT"
    run_in_dir "$DEEP_ROOT" env \
      "NUNCHAKU_REPO_DIR=${NUNCHAKU_ROOT}" \
      "PYTHONPATH=${NUNCHAKU_ROOT}:${DEEP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_CMD[@]}" \
      -m examples.diffusion.scripts.convert_kitchen_native \
      --raw-nunchaku "$RAW_NUNCHAKU" \
      --base-comfy "$BASE_COMFY" \
      --output "$OUTPUT" \
      "${awq_args[@]}"
  fi
}

validate_final_output() {
  [[ "$VALIDATE_OUTPUT" == true ]] || return 0
  if [[ "$DRY_RUN" == true ]]; then
    log "dry-run: skip final validation"
    return 0
  fi
  require_file "$OUTPUT"
  log "validating final safetensors: $OUTPUT"
  PYTHONPATH="${COMFY_QUANTS_ROOT}/src:${DEEP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_CMD[@]}" - "$OUTPUT" "$HASH_OUTPUT" <<'PY'
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
from safetensors import safe_open

path = Path(sys.argv[1])
hash_output = sys.argv[2].lower() == "true"
svd = 0
awq = 0
bad_layout = []
formats: dict[str, int] = {}
with safe_open(path, framework="pt", device="cpu") as f:
    meta = f.metadata() or {}
    keys = list(f.keys())
    for key in keys:
        if not key.endswith(".comfy_quant"):
            continue
        tensor = f.get_tensor(key)
        try:
            conf = json.loads(bytes(tensor.tolist()).decode("utf-8"))
        except Exception:
            continue
        fmt = str(conf.get("format", ""))
        formats[fmt] = formats.get(fmt, 0) + 1
        if fmt == "svdquant_w4a4":
            svd += 1
            if conf.get("layout") != "kitchen_tile_packed_w4a4":
                bad_layout.append(key)
        elif fmt == "awq_w4a16":
            awq += 1

report = {
    "path": str(path),
    "size_bytes": path.stat().st_size,
    "metadata_svdquant_storage_layout": meta.get("svdquant_storage_layout"),
    "metadata_awq_modulation_layout": meta.get("awq_modulation_layout"),
    "tensor_count": len(keys),
    "quant_formats": formats,
    "svdquant_w4a4_layers": svd,
    "awq_w4a16_layers": awq,
    "bad_svdquant_layout_keys": bad_layout[:20],
}
if hash_output:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    report["sha256"] = h.hexdigest()
print(json.dumps(report, ensure_ascii=False, indent=2))
if svd == 0:
    raise SystemExit("validation failed: no svdquant_w4a4 comfy_quant entries found")
if bad_layout:
    raise SystemExit("validation failed: some svdquant_w4a4 layers are not kitchen_tile_packed_w4a4")
PY
}

preflight_common
if [[ "$ROUTE" == "nunchaku-bridge" ]]; then
  preflight_nunchaku
elif [[ -z "$OUTPUT" ]]; then
  die "--output is required"
fi

log "configuration:"
log "  route              = $ROUTE"
log "  deepcompressor     = $DEEP_ROOT"
log "  nunchaku           = $NUNCHAKU_ROOT"
log "  model_id           = $MODEL_ID"
log "  candidate/gpus     = $CANDIDATE / $GPUS"
log "  runs_root          = $RUNS_ROOT"
log "  export_root        = $EXPORT_ROOT"
log "  export_name        = $EXPORT_NAME"
log "  quant_path         = ${QUANT_PATH:-<will be produced/resolved>}"
log "  base_comfy         = ${BASE_COMFY:-<not used>}"
log "  raw_nunchaku       = $RAW_NUNCHAKU"
log "  output             = $OUTPUT"

run_ptq_if_needed
if [[ "$DRY_RUN" != true ]]; then
  validate_quant_path "$QUANT_PATH"
fi

if [[ "$ROUTE" == "deepcompressor-import" ]]; then
  run_deepcompressor_import_route
else
  run_nunchaku_bridge_route
fi

validate_final_output
log "done: $OUTPUT"
