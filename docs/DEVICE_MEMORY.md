# Device memory model (VTL autograd + CUDA)

Status: **Phase 1** complete · **Phase 2** done (`VTL_GPU_ACTIVATIONS=1`, #101/#104) ·
**Phase 3** opt-in (`VTL_CUDA_BACKWARD=1`) — Linear backward GEMMs on GPU when enabled.

## Policy

| Stage | Where data lives | Why |
|-------|------------------|-----|
| Parameters (weights, bias) | CPU (`CpuStorage`) | Optimizers and gates use CPU matmul today |
| Forward (Linear/Conv2D) | Compute on GPU when `VTL_USE_CUDA=1` | cuBLAS/cuDNN |
| Forward output | **CPU tensor** | `Variable` and gates expect `CpuStorage` |
| Backward | CPU by default; GPU GEMM for Linear when `VTL_CUDA_BACKWARD=1` | Conv2D backward still on host |
| Optimizer (Adam) | CPU by default; GPU moments when `VTL_CUDA_OPTIMIZER=1` | Parameter sqrt step on CPU |
| Optimizer step | CPU | unchanged |

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

Build: `-d cuda` required for GPU code paths.

## API

```v
import vtl.autograd
import vtl.nn.models

mut ctx := autograd.ctx[f64]()
// ctx.device_session is initialized; reuses buffers across forwards on this ctx

mut model := models.sequential_from_ctx[f64](ctx)
// ... train — Linear layers use session when f64 + CUDA enabled
```

## Roadmap (issue #91 follow-ups)

- **Phase 2**: GPU-resident `Variable` (`gpu_activation`, #101) — done (#104)
- **Phase 3**: CUDA backward for Linear (opt-in) — done; Conv2D backward still CPU
- **Phase 4**: Adam moment updates on GPU (#106, `VTL_CUDA_OPTIMIZER=1`); device-resident m/v + fused sqrt TBD

See [DEV_LIGHTWEIGHT.md](DEV_LIGHTWEIGHT.md) for safe test commands.
