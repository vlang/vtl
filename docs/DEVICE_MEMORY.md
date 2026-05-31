# Device memory model (VTL autograd + CUDA)

Status: **Phase 1** — opt-in CUDA compute with CPU-resident activations for autograd.

## Policy

| Stage | Where data lives | Why |
|-------|------------------|-----|
| Parameters (weights, bias) | CPU (`CpuStorage`) | Optimizers and gates use CPU matmul today |
| Forward (Linear/Conv2D) | Compute on GPU when `VTL_USE_CUDA=1` | cuBLAS/cuDNN |
| Forward output | **CPU tensor** | `Variable` and gates expect `CpuStorage` |
| Backward | CPU | `LinearGate` / `conv2d_backward` use `vtl.la` on host |
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
| `VTL_TEST_CUDA=1` | Run GPU tests |

Build: `-d cuda` required for GPU code paths.

## API

```v
mut ctx := autograd.ctx[f64]()
// ctx.device_session is initialized; reuses buffers across forwards on this ctx

mut model := models.sequential_from_ctx[f64](ctx)
// ... train — Linear layers use session when f64 + CUDA enabled
```

## Roadmap (issue #91 follow-ups)

- **Phase 2**: GPU-resident `Variable` for forward-only tensors (sum-type storage)
- **Phase 3**: CUDA backward for Linear / Conv2D
- **Phase 4**: Optimizer state on device (fused Adam step)

See [DEV_LIGHTWEIGHT.md](DEV_LIGHTWEIGHT.md) for safe test commands.
