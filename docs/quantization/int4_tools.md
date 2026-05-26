# INT4 artifact tools

These commands work with existing INT4 artifacts. They do not run calibration,
search, or GPTQ.

## Inspect a tile-pack artifact

`inspect-int4` validates the static structure of a `svdquant_w4a4` tile-pack checkpoint.

```bash
comfy-quants inspect-int4 \
  --artifact /absolute/path/to/qwen_edit_2511_int4_tilepack.safetensors \
  --family qwen_image_edit \
  --format svdquant_w4a4 \
  --strict-qwen-image-edit-2511 \
  --json
```

The inspector checks tensor names, tensor counts, QKV split layout, rank, low-rank
branch tensors, and Qwen-Image-Edit-2511 shape rules. It does not run image inference.

## Repack an existing INT4 artifact

`export-int4` converts already-quantized inputs into the kitchen tile-pack file layout.

```bash
comfy-quants export-int4 \
  --format svdquant_w4a4 \
  --source-format deepcompressor-qwen-image-edit \
  --source /absolute/path/to/deepcompressor-ptq-artifacts \
  --out runs/qwen-edit-2511/export-int4 \
  --device cuda:0 \
  --hash-output \
  --json
```

Directory output:

```text
diffusion_pytorch_model.svdquant_w4a4.safetensors
```

Use [`qwen_image_edit_2511_int4.md`](qwen_image_edit_2511_int4.md) when calibration,
search, PTQ, conversion, and tile-pack export should run as one flow.

## Format references

- [`../formats/svdquant_w4a4_kitchen_tilepack.md`](../formats/svdquant_w4a4_kitchen_tilepack.md)
- [`../formats/awq_w4a16.md`](../formats/awq_w4a16.md)
