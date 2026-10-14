# INT8 quantization

`vtl.quantization` provides opt-in affine signed INT8 conversion for f32/f64
tensors. Quantization parameters use

```text
q = clamp(round_to_even(x / scale + zero_point), qmin, qmax)
x = (q - zero_point) * scale
```

The initial API includes per-tensor and per-channel parameters, finite-value
min/max calibration observers, and explicit f32/f64 quantize/dequantize entry
points.
Quantized results are ordinary `Tensor[i8]` values. This is a storage and
calibration foundation; it does not yet provide integer Linear/Conv kernels,
layer conversion, accelerator dispatch, or a performance claim. Those remain
tracked by [issue #792](https://github.com/vlang/vtl/issues/792).

See the runnable [INT8 quantization example](../examples/quantization_int8/main.v).

Per-channel conversion accepts an axis and one scale/zero point per channel.
Negative axes are normalized against the tensor rank. Conversion traverses
logical tensor order, including strided views, and returns a row-major tensor
with the same shape. Non-finite calibration or input values are errors rather
than silently saturating NaNs or infinities.

## Validation

Run from `~/.vmodules` with a local memory cap:

```sh
VJOBS=2 systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- v test ./vtl/quantization
```
