# VTL — ML Roadmap & Launch Tracking

Central planning lives on the **vlang org project board**:

**https://github.com/orgs/vlang/projects/8** (Vlang ML Roadmap — VSL + VTL)

Detailed file-level roadmap: [ROADMAP.md](../ROADMAP.md) in this repo.

## Critical path (open issues)

| Priority | Issue | Topic |
|----------|-------|--------|
| P0 | [#89](https://github.com/vlang/vtl/issues/89) | Wire `LinearLayer` to CUDA |
| P0 | [#90](https://github.com/vlang/vtl/issues/90) | Wire `Conv2D` to `vsl.cuda` |
| P0 | [#91](https://github.com/vlang/vtl/issues/91) | Device-resident autograd |
| P1 | [#88](https://github.com/vlang/vtl/issues/88) | vs NumPy/PyTorch benchmarks |
| P1 | [#87](https://github.com/vlang/vtl/issues/87) | CIFAR checkpoint example |
| P1 | [#41](https://github.com/vlang/vtl/issues/41) | Windows example crash |

VSL-side blockers: [vlang/vsl#280](https://github.com/vlang/vsl/issues/280),
[#281](https://github.com/vlang/vsl/issues/281), [#282](https://github.com/vlang/vsl/issues/282).

## Local development

Work from `~/.vmodules` (see [vlang pack](https://github.com/vlang/vtl)):

```bash
v up
cd ~/.vmodules
# Smoke (avoid full v test vtl on low-RAM machines)
v test vtl/nn vtl/datasets
v run vtl/examples/nn_cifar10_tiny_synth/main.v
# CUDA training (when wired — issue #89+)
# v -d cuda run vtl/examples/nn_cifar10_tiny/main.v
```

## CI recommendation

- **Default PR:** `nn_cifar10_tiny_synth` + scoped `v test vtl/nn`
- **Full suite / CUDA:** labeled runners only
