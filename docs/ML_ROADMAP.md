# VTL — ML Roadmap & Launch Tracking

Maintainer planning: **https://github.com/orgs/vlang/projects/8** (Vlang ML Roadmap — VSL + VTL; may require project access)

Detailed roadmap: [ROADMAP.md](../ROADMAP.md)

GPU memory: [DEVICE_MEMORY.md](DEVICE_MEMORY.md)

## Completed issues

| Issue | Topic |
|-------|--------|
| [#87](https://github.com/vlang/vtl/issues/87) | Serialization + CIFAR checkpoints |
| [#88](https://github.com/vlang/vtl/issues/88) | vs NumPy/PyTorch benchmarks + PR comments |
| [#89](https://github.com/vlang/vtl/issues/89)–[#91](https://github.com/vlang/vtl/issues/91) | CUDA Linear/Conv2D + `DeviceSession` (Phase 1) |
| [#101](https://github.com/vlang/vtl/issues/101)/[#104](https://github.com/vlang/vtl/pull/104) | GPU activation chain (Phase 2) |
| [#105](https://github.com/vlang/vtl/pull/105) | Linear CUDA backward (`VTL_CUDA_BACKWARD=1`, Phase 3) |
| [#107](https://github.com/vlang/vtl/issues/107) | Conv2D CUDA backward (cuDNN, same eligibility as forward) |
| [#111](https://github.com/vlang/vtl/pull/111)/[#114](https://github.com/vlang/vtl/pull/114) | Adam on GPU + persistent `DeviceSession` slots (#106) |
| [#110](https://github.com/vlang/vtl/issues/110) | Vulkan f32 Linear in `Sequential` forward (`VTL_USE_VULKAN=1`, `-d vulkan`) |
| [#116](https://github.com/vlang/vtl/issues/116) | f32 autograd: `Sequential` + MSE forward/backprop compile |
| — | f32 tiny training: `nn_cifar10_f32_tiny_synth` + `f32_training_smoke_test` |
| — | CUDA training smoke: `nn_cifar10_cuda` + `nn/cuda_training_smoke_test` |
| — | f32 Vulkan training: `nn_cifar10_vulkan` + `f32_vulkan_training_smoke_d_vulkan_test` |
| — | Vulkan Conv2D f32 forward/backward (same-padding, im2col+GEMM) |
| — | Vulkan ReLU/Sigmoid/Softplus/SELU/HardSwish f32; experimental per-layer dispatch with CPU fallback |
| — | Vulkan Adam f32 fused shader (`VTL_USE_VULKAN=1`, VSL `adam_step`) |
| — | Conv2D autograd: register weight/bias parents (`conv2d_autograd_smoke_test`) |
| [#86](https://github.com/vlang/vtl/issues/86) | `DataLoader` |
| [#148](https://github.com/vlang/vtl/issues/148) | Contribution standards and test conventions |
| [#152](https://github.com/vlang/vtl/issues/152) | NAdam and RAdam optimizers |
| [#153](https://github.com/vlang/vtl/issues/153) | L1/MAE, Hinge, and Focal losses |
| [#154](https://github.com/vlang/vtl/issues/154) | GRU layer |
| [#155](https://github.com/vlang/vtl/issues/155) | Conv1D layer |
| [#157](https://github.com/vlang/vtl/issues/157) | Expanded Windows test coverage |
| [#158](https://github.com/vlang/vtl/issues/158) | Gradient-check utility |
| [#159](https://github.com/vlang/vtl/issues/159) | Clamp/clip autograd gate |
| [#160](https://github.com/vlang/vtl/issues/160) | Reshape/transpose/concat backward gates |
| [#162](https://github.com/vlang/vtl/issues/162) | Runnable OpenCL/VCL CI example |
| [#163](https://github.com/vlang/vtl/issues/163) | Transformer/attention end-to-end example |
| — | `from_array` clones shape (fixes [#41](https://github.com/vlang/vtl/issues/41) aliasing) |

**VSL (downstream):** [#280](https://github.com/vlang/vsl/issues/280)–[#285](https://github.com/vlang/vsl/issues/285), [#304](https://github.com/vlang/vsl/pull/304) conv2d backward GEMM, [#305](https://github.com/vlang/vsl/pull/305) Adam shaders.

## Current open issues (checked 2026-10-07)

| Priority | Issue | Topic |
|----------|-------|--------|
| P1 | [#161](https://github.com/vlang/vtl/issues/161) | CUDA backward for LSTM, Attention, BatchNorm, LayerNorm, Embedding, pooling, and remaining Dropout training paths; runtime validation is still required |
| P2 | [#63](https://github.com/vlang/vtl/issues/63) | ARM GPU support |
| P2 | [#40](https://github.com/vlang/vtl/issues/40) | YOLO/fused autograd gates; requires benchmark evidence |
| P2 | [#3](https://github.com/vlang/vtl/issues/3) | Evaluate compiler aliasing support without unsafe assumptions |
| Research | [#52](https://github.com/vlang/vtl/issues/52) | Compare Burn capabilities and architecture |

Project #8 currently contains #3, #40, #52, and #63, but omits the open
CUDA-backward issue #161. The project board and repository issue inventory need
reconciliation. Issue #41 is closed and is no longer a beta gate.

## Additional post-beta tracking

| Priority | Issue | Topic |
|----------|-------|--------|
| P2 | — | Vulkan: persistent GPU activation chain between layers (CUDA has `VTL_GPU_ACTIVATIONS`) |

## Performance engineering backlog

The historical [#64 performance engineering proposal](https://github.com/vlang/vtl/issues/64)
is closed, but its acceptance checklist is not complete in the current tree.
VTL has specific optimized paths, including fused Vulkan Adam, while the
broader work remains: general fused operation chains, FP16/BF16 and automatic
mixed precision, reusable GPU memory pools, asynchronous Vulkan execution,
backend profiling, and checkpointing. Treat these as future work until each
has an implementation, correctness coverage, and benchmark evidence; issue
closure alone is not completion evidence.

The beta contract is tensors, autograd, high-level layers/losses/optimizers,
datasets, and f32 CPU training. CUDA and Vulkan remain opt-in experimental
backend paths during beta. Generic f64 training is tracked as post-beta until
the autograd `Gate[T]` interface compiles without mixed `Payload[f32]` and
`Payload[f64]` specialization.

## Local development

**Lightweight workflow:** [DEV_LIGHTWEIGHT.md](DEV_LIGHTWEIGHT.md)

```bash
v up
cd ~/.vmodules
v test ./vtl/nn ./vtl/datasets
v run ./vtl/examples/nn_cifar10_tiny_synth/main.v
v run ./vtl/examples/nn_cifar10_f32_tiny_synth/main.v
# Vulkan f32 full stack (use -prod for GPU)
# VTL_USE_VULKAN=1 v -prod -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
# VTL_USE_VULKAN=1 VTL_TEST_VULKAN=1 VJOBS=1 v -prod -d vulkan test vtl/nn/f32_vulkan_training_smoke_d_vulkan_test.v
```

## CI

- **Default PR:** `bin/test` (scoped modules + `nn_cifar10_tiny_synth`)
- **f32 beta smoke:** `nn_cifar10_f32_tiny_synth` and f32 autograd/training tests
- **`full-ml` label:** `ci-full-ml.yml` runs `bin/test --full`
- **CUDA / Vulkan:** local or future labeled workflows

## Project board sync

From repo root (requires `gh` scopes `read:project`, `project`):

```bash
./.github/scripts/sync-ml-project-8.sh
```
