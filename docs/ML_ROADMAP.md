# VTL — ML Roadmap & Launch Tracking

Central planning lives on the **vlang org project board**:

**https://github.com/orgs/vlang/projects/8** (Vlang ML Roadmap — VSL + VTL)

Detailed file-level roadmap: [ROADMAP.md](../ROADMAP.md) in this repo.

## Critical path (open issues)

| Priority | Issue | Topic |
|----------|-------|--------|
| P1 | [#41](https://github.com/vlang/vtl/issues/41) | Windows example crash |
| — | — | `nn_cifar10_cuda` example added (opt-in GPU smoke) |
| P2 | follow-up | Benchmark PR comments + PyTorch baselines (#88 scripts done) |

**Done (2026-05-31):** #89–#91 (PR #93), #88 (PR #94), #87 (PR #95).  
**VSL done:** #280, #281, #282.

## Local development

**Lightweight workflow (avoid OOM):** [DEV_LIGHTWEIGHT.md](DEV_LIGHTWEIGHT.md)

Work from `~/.vmodules` (see [vlang pack](https://github.com/vlang/vtl)):

```bash
v up
cd ~/.vmodules
# Smoke (avoid full v test vtl on low-RAM machines)
v test vtl/nn vtl/datasets
v run vtl/examples/nn_cifar10_tiny_synth/main.v
# CUDA Linear/Conv2D (opt-in, PR #93)
# VTL_USE_CUDA=1 v -d cuda run vtl/examples/nn_cifar10_tiny_synth/main.v
# VTL_USE_CUDA=1 v -d cuda test vtl/nn/layers/linear_cuda_test.v
```

## CI recommendation

- **Default PR:** `nn_cifar10_tiny_synth` + scoped `v test vtl/nn`
- **Full suite / CUDA:** labeled runners only
