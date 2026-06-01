# Device memory model (VTL autograd + GPU backends)

Status: **Phase 1** complete · **Phase 2** done (`VTL_GPU_ACTIVATIONS=1`, #101/#104) ·
**Phase 3** done (`VTL_CUDA_BACKWARD=1`) · **Phase 4** done
(`VTL_CUDA_OPTIMIZER=1`) · Vulkan f32 training path done (`VTL_USE_VULKAN=1`).

## Policy

| Stage | Where data lives | Why |
|-------|------------------|-----|
| Parameters (weights, bias) | CPU (`CpuStorage`) | Stable serialization/autograd interface |
| Forward (Linear/Conv2D) | Compute on GPU when enabled | CUDA cuBLAS/cuDNN or Vulkan GEMM/im2col |
| Forward output | **CPU tensor** | `Variable` and gates expect `CpuStorage` |
| Backward | CPU by default; GPU when eligible | CUDA Linear/Conv2D; Vulkan Linear/Conv2D `d_weight` |
| Optimizer (Adam) | CPU by default; GPU when enabled | CUDA persistent slots; Vulkan fused f32 shader |
| Optimizer step | CPU sync result | Host tensors remain canonical |

Sync points (host ↔ device) per Linear forward today:

1. Upload `x`, `W` (via flat buffers in `DeviceSession`)
2. Download GEMM result
3. Bias add on CPU

Phase 1 removes redundant **allocations** via `DeviceSession` buffer reuse on the same `Context`.

## Environment

| Variable | Effect |
|----------|--------|
| `VTL_USE_CUDA=1` | Enable GPU forward for eligible ops |
| `VTL_GPU_ACTIVATIONS=1` | Phase 2: chain GPU activations across Linear layers |
| `VTL_CUDA_BACKWARD=1` | Phase 3: cuBLAS GEMM for Linear gate backward |
| `VTL_CUDA_OPTIMIZER=1` | Phase 4: cuBLAS moment updates for Adam |
| `VTL_TEST_CUDA=1` | Run GPU tests |
| `VTL_USE_VULKAN=1` | f32 Linear/Conv2D/ReLU/Sigmoid/Adam via Vulkan |
| `VTL_TEST_VULKAN=1` | Run Vulkan integration tests |

Build: `-d cuda` or `-d vulkan` required for GPU code paths. Use `v -prod`
with Vulkan on machines where debug instance creation is unstable.

## API

```v
import vtl.autograd
import vtl.autograd_cuda
import vtl.nn.models

mut ctx := autograd.ctx[f64]()
autograd_cuda.attach_context_session(mut ctx)
// ctx.device_session reuses buffers across forwards on this ctx

mut model := models.sequential_from_ctx[f64](ctx)
// ... train — Linear layers use session when f64 + CUDA enabled
```

## Roadmap (issue #91 follow-ups)

- **Phase 2**: GPU-resident `Variable` (`gpu_activation`, #101) — done (#104)
- **Phase 3**: CUDA backward for Linear + Conv2D (opt-in `VTL_CUDA_BACKWARD`) — done (#107)
- **Phase 4**: Adam on GPU with persistent m/v/θ in `DeviceSession` — done

See [DEV_LIGHTWEIGHT.md](DEV_LIGHTWEIGHT.md) for safe test commands.

## Vulkan (f32, opt-in)

| Variable | Effect |
|----------|--------|
| `VTL_USE_VULKAN=1` | GPU forward/backward for Linear, Conv2D (same-padding), ReLU/Sigmoid, Adam |
| `VTL_TEST_VULKAN=1` | Run Vulkan integration tests |

Build: `-d vulkan` and `v -prod` for GPU execution. Tensors remain
CPU-backed; ops sync via host buffers (same policy as CUDA Phase 1).

Current Vulkan f32 stack:

| Component | GPU path |
|-----------|----------|
| Linear forward/backward | VSL Vulkan GEMM |
| Conv2D forward/backward | VSL Vulkan im2col + GEMM (same-padding) |
| ReLU/Sigmoid | VSL Vulkan compute shaders |
| Adam | VSL Vulkan fused `adam_step` shader |
