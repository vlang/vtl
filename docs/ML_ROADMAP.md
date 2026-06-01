# VTL — ML Roadmap & Launch Tracking

Central planning: **https://github.com/orgs/vlang/projects/8** (Vlang ML Roadmap — VSL + VTL)

Detailed roadmap: [ROADMAP.md](../ROADMAP.md)

GPU memory: [DEVICE_MEMORY.md](DEVICE_MEMORY.md)

## Done (2026-06-01)

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
| — | f32 Vulkan training: `nn_cifar10_vulkan` + `f32_vulkan_training_smoke_test` |
| — | Vulkan Conv2D f32 forward/backward (same-padding, im2col+GEMM) |
| — | Vulkan ReLU/Sigmoid f32 (`relu_vulkan_f32` / `sigmoid_vulkan_f32`) |
| — | Vulkan Adam f32 fused shader (`VTL_USE_VULKAN=1`, VSL `adam_step`) |
| — | Conv2D autograd: register weight/bias parents (`conv2d_autograd_smoke_test`) |
| [#86](https://github.com/vlang/vtl/issues/86) | `DataLoader` |
| — | `from_array` clones shape (fixes [#41](https://github.com/vlang/vtl/issues/41) aliasing) |

**VSL (downstream):** [#280](https://github.com/vlang/vsl/issues/280)–[#285](https://github.com/vlang/vsl/issues/285), [#304](https://github.com/vlang/vsl/pull/304) conv2d backward GEMM, [#305](https://github.com/vlang/vsl/pull/305) Adam shaders.

## Critical path (open)

| Priority | Issue | Topic |
|----------|-------|--------|
| P1 | [#41](https://github.com/vlang/vtl/issues/41) | Windows example crash — shape clone landed; needs Windows CI confirmation |
| P2 | [#63](https://github.com/vlang/vtl/issues/63) | ARM GPU support |
| P2 | — | `v check-md -hide-warnings` on remaining docs (example READMEs done) |
| P2 | — | Vulkan: persistent GPU activation chain between layers (CUDA has `VTL_GPU_ACTIVATIONS`) |

## Local development

**Lightweight workflow:** [DEV_LIGHTWEIGHT.md](DEV_LIGHTWEIGHT.md)

```bash
v up
cd ~/.vmodules
v test vtl/nn vtl/datasets
v run vtl/examples/nn_cifar10_tiny_synth/main.v
v run vtl/examples/nn_cifar10_f32_tiny_synth/main.v
# Vulkan f32 full stack (use -prod for GPU)
# VTL_USE_VULKAN=1 v -prod -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
```

## CI

- **Default PR:** `bin/test` (scoped modules + `nn_cifar10_tiny_synth`)
- **`full-ml` label:** `ci-full-ml.yml` runs `bin/test --full`
- **CUDA / Vulkan:** local or future labeled workflows

## Project board sync

From repo root (requires `gh` scopes `read:project`, `project`):

```bash
./.github/scripts/sync-ml-project-8.sh
```
