# INT8 quantization

`vtl.quantization` provides opt-in affine signed INT8 conversion for f32/f64
tensors. Quantization parameters use

```text
q = clamp(round_to_even(x / scale + zero_point), qmin, qmax)
x = (q - zero_point) * scale
```

The API includes per-tensor and per-channel parameters, finite-value min/max
calibration observers, and explicit f32/f64 quantize/dequantize entry points.
Quantized results are ordinary `Tensor[i8]` values. `linear_int8_f32` provides
a CPU inference path with per-tensor activation quantization, per-output-channel
weight quantization, exact i64 accumulation, and f32 output with bias. Linear
inference uses VSL's AVX2-dispatched INT16 dot product for reductions of at
least 128 features on supported x86 CPUs, with a scalar fallback elsewhere.
CUDA builds also provide `linear_int8_f32_cuda(dev, ...)`, an explicit cuBLAS
path that returns a CPU-resident f32 tensor. It pads GEMM dimensions and keeps
integer accumulation exact, including reductions larger than INT32. The current
API uploads and transposes weights per call; this makes it slower than CPU INT8
for small inference workloads, so it remains experimental pending reusable
device-resident weights and broader benchmarks.
Quantized Conv2D is available through `conv2d_int8_f32` for NCHW/OIHW tensors, including
groups, padding, stride, and dilation via `QuantizedConv2DConfig`. Layer
conversion helpers `quantize_linear_layer_f32` and
`quantize_conv2d_layer_f32` create forward-only inference wrappers from trained
f32 weight/bias tensors and externally calibrated activation parameters.
CUDA INT8 Linear is an experimental explicit path. GPU Conv2D, reusable
device-resident quantized weights, broader model conversion, and performance
parity remain open under [issue #792](https://github.com/vlang/vtl/issues/792).

See the runnable [INT8 quantization example](../examples/quantization_int8/main.v).
The end-to-end f32-vs-INT8 latency and weight-payload benchmark is
`../benchmarks/quantization_int8_bench.v`; compile with `-prod` from
`~/.vmodules` and run the resulting binary separately under the memory limit
documented in `docs/DEV_LIGHTWEIGHT.md`.

The benchmark runs `[32, 256] x [512, 256]` with two warmups and eight timed
calls per process. It includes activation quantization and output allocation in
the INT8 timing, while weights are quantized before timing as they would be for
inference. It reports the maximum absolute difference from the f32 result and
weight tensor payload plus per-channel metadata; bias and V object overhead are
excluded. On an AMD Ryzen 9 5900X with V 0.5.2 (`7f5b9a3`) and GCC 16.2.1,
three independent runs measured a median f32/INT8 latency ratio of 6.792x
(per-run ratios 6.521x-6.864x), with maximum absolute error 0.015401. Weight
payload was 524,288 bytes for f32 versus 131,072 bytes plus 8,192 bytes of
per-channel metadata for INT8 (3.765x smaller including that metadata). This is
one CPU, shape, and implementation only; it is not evidence of general PyTorch,
NumPy, or accelerator performance parity.

The opt-in CUDA benchmark is
`../benchmarks/quantization_int8_cuda_bench.v`. On an NVIDIA GeForce RTX 3060
with V 0.5.2, three independent production runs of the same `[32, 256] x
[512, 256]` workload measured median f32 latency 21.1657 ms, CPU INT8 latency
0.7157 ms, and CUDA INT8 latency 0.8749 ms. CUDA's maximum absolute error
matched CPU INT8 at 0.015401, but the per-call transfers made it 1.22x slower
than CPU INT8 on this small workload. These results are specific to this
machine and shape; they do not establish a general GPU speedup.

The same benchmark accepts optional `batch in_features out_features` arguments.
Across seven production runs of `[128, 512] x [1024, 512]`, median f32, CPU
INT8, and CUDA INT8 latencies were 56.467 ms, 6.7714 ms, and 4.4597 ms,
respectively; CUDA was 1.52x faster than CPU INT8 at the median with the same
0.021986 maximum absolute error. CUDA timings ranged from 4.3181 to 9.7551 ms,
so these host measurements show workload scaling, not a universal speedup.

Run reproducibly from `~/.vmodules`:

```sh
systemd-run --user --scope --working-directory="$HOME/.vmodules" \
  -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=200% -E VJOBS=1 -- \
  v -no-parallel -prod -o /tmp/vtl-quantization-int8-bench \
  ./vtl/benchmarks/quantization_int8_bench.v
systemd-run --user --scope --working-directory="$HOME/.vmodules" \
  -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=200% -- \
  /tmp/vtl-quantization-int8-bench
```

The Conv2D benchmark is `../benchmarks/quantization_int8_conv2d_bench.v`. It
uses input `[1, 8, 32, 32]`, weights `[16, 8, 3, 3]`, padding 1, one warmup,
and three timed calls. The CPU path packs each input patch once and reuses it
across output channels. On the same Ryzen 9 5900X / V 0.5.2 / GCC 16.2.1 host,
three independent runs had median f32 latency 0.8562 ms and INT8 latency
0.9818 ms (median per-run f32/INT8 ratio 0.857x; range 0.769x-0.924x), so
INT8 remained slower on this shape. Maximum absolute error was 0.038690.
Effective INT8 weight payload plus per-channel metadata was 1,408 bytes versus
4,608 bytes f32 (3.273x smaller, excluding bias and V object overhead). This
reference CPU kernel is currently a storage/accuracy path, not a Conv2D speedup.

```sh
systemd-run --user --scope --working-directory="$HOME/.vmodules" \
  -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=200% -E VJOBS=1 -- \
  v -no-parallel -prod -o /tmp/vtl-quantization-int8-conv2d-bench \
  ./vtl/benchmarks/quantization_int8_conv2d_bench.v
systemd-run --user --scope --working-directory="$HOME/.vmodules" \
  -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=200% -- \
  /tmp/vtl-quantization-int8-conv2d-bench
```

Run the CUDA benchmark from `~/.vmodules`:

```sh
systemd-run --user --scope --working-directory="$HOME/.vmodules" \
  -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=200% -E VJOBS=1 -- \
  v -d cuda -no-parallel -prod -o /tmp/vtl-quantization-int8-cuda-bench \
  ./vtl/benchmarks/quantization_int8_cuda_bench.v
systemd-run --user --scope --working-directory="$HOME/.vmodules" \
  -p MemoryMax=2G -p MemorySwapMax=0 -p CPUQuota=200% -- \
  /tmp/vtl-quantization-int8-cuda-bench 128 512 1024
```

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
