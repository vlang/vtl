# VTL — ML Roadmap & Launch Tracking

Central planning: **https://github.com/orgs/vlang/projects/8** (Vlang ML Roadmap — VSL + VTL)

Detailed roadmap: [ROADMAP.md](../ROADMAP.md)

GPU memory: [DEVICE_MEMORY.md](DEVICE_MEMORY.md)

## Done (2026-05-31)

| Issue | Topic |
|-------|--------|
| [#87](https://github.com/vlang/vtl/issues/87) | Serialization + CIFAR checkpoints |
| [#88](https://github.com/vlang/vtl/issues/88) | vs NumPy/PyTorch benchmarks + PR comments |
| [#89](https://github.com/vlang/vtl/issues/89)–[#91](https://github.com/vlang/vtl/issues/91) | CUDA Linear/Conv2D + `DeviceSession` (Phase 1) |
| [#101](https://github.com/vlang/vtl/issues/101)/[#104](https://github.com/vlang/vtl/pull/104) | GPU activation chain (Phase 2) |
| Phase 3 | Linear CUDA backward (`VTL_CUDA_BACKWARD=1`) |
| [#86](https://github.com/vlang/vtl/issues/86) | `DataLoader` |

**VSL (downstream):** [#280](https://github.com/vlang/vsl/issues/280)–[#285](https://github.com/vlang/vsl/issues/285).

## Critical path (open)

| Priority | Issue | Topic |
|----------|-------|--------|
| P1 | [#41](https://github.com/vlang/vtl/issues/41) | Windows example crash |
| P1 | — | Full `nn_cifar10` in CI (compile OOM on default runners) |
| P1 | follow-up | GPU Phase 4 optimizer on device ([DEVICE_MEMORY.md](DEVICE_MEMORY.md)) |
| P2 | — | Conv2D CUDA backward; Vulkan in `Sequential` training |
| P2 | [#63](https://github.com/vlang/vtl/issues/63) | ARM GPU support |
| P2 | — | Vulkan in `Sequential` training (today: smoke in `nn_cifar10_vulkan`) |
| P2 | — | `v check-md -hide-warnings` on all example READMEs |

## Local development

**Lightweight workflow:** [DEV_LIGHTWEIGHT.md](DEV_LIGHTWEIGHT.md)

```bash
v up
cd ~/.vmodules
v test vtl/nn vtl/datasets
v run vtl/examples/nn_cifar10_tiny_synth/main.v
# CUDA (opt-in)
# VTL_USE_CUDA=1 v -d cuda run vtl/examples/nn_cifar10_cuda/main.v
# Vulkan smoke
# v -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
```

## CI recommendation

- **Default PR:** `nn_cifar10_tiny_synth` + scoped `v test vtl/nn`
- **Full / CUDA / Vulkan:** labeled runners or local only

## Project board sync

From repo root (requires `gh` scopes `read:project`, `project`):

```bash
./.github/scripts/sync-ml-project-8.sh
```
