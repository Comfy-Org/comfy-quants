# Comfy Quants documentation

Start with the quantization format you want to produce. Format pages link to the
model-family flows currently implemented for that format.

## Start here

| Goal | Page |
| --- | --- |
| Choose a quantization format | [`quantization/README.md`](quantization/README.md) |
| Export FP8 checkpoints | [`quantization/fp8.md`](quantization/fp8.md) |
| Export INT8 W8A8 (+ConvRot) checkpoints | [`quantization/int8_w8a8.md`](quantization/int8_w8a8.md) |
| Export MXFP8 (microscaling, Blackwell) checkpoints | [`quantization/mxfp8.md`](quantization/mxfp8.md) |
| Export INT4 tile-pack checkpoints | [`quantization/int4.md`](quantization/int4.md) |
| Look up command syntax | [`cli.md`](cli.md) |
| Understand repository architecture | [`architecture.md`](architecture.md) |

## Format references

| Format | Reference |
| --- | --- |
| FP8 E4M3 / E5M2 | [`formats/fp8.md`](formats/fp8.md) |
| INT8 W8A8 (+ConvRot) | [`formats/int8_w8a8.md`](formats/int8_w8a8.md) |
| MXFP8 microscaling | [`formats/mxfp8.md`](formats/mxfp8.md) |
| SVDQuant W4A4 kitchen tile-pack | [`formats/svdquant_w4a4_kitchen_tilepack.md`](formats/svdquant_w4a4_kitchen_tilepack.md) |
| AWQ W4A16 | [`formats/awq_w4a16.md`](formats/awq_w4a16.md) |

## ComfyUI integration model

Run quantization with this package, then load the exported artifact with a
compatible ComfyUI setup. Downstream custom-node projects can depend on
`comfy-quants` and call its CLI or Python API when they need an in-UI workflow.
