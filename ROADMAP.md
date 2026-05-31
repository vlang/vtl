# VTL Neural Network Roadmap

> **Goal**: Production-grade Neural Network support in VTL with CUDA/GPU
> acceleration, DataLoader, CIFAR-10 example, and model serialization.

**Reposity**: [vlang/vtl](https://github.com/vlang/vtl) · **VSL Backend**: [vlang/vsl](https://github.com/vlang/vsl)

---

## ✅ Completed

### Phase 0 — Foundation
- [x] Tensor core (`vtl.Tensor`), creation, slicing, broadcasting, map/reduce
- [x] Autograd engine (`Variable`, `Context`, reverse-mode backprop)
- [x] Linear Algebra via VSL (`matmul`, `solve`, `lstsq`, `qr`, `lu`, `cholesky`, `pinv`)
- [x] NN layers: Linear, Conv2D, MaxPool, Flatten, LSTM, LayerNorm, Attention
- [x] Losses: MSE, BCE, CrossEntropy, Huber
- [x] Optimizers: Adam, AdamW, RMSProp, AdaGrad, SGD + schedulers

### Phase 1 — Vulkan Compute Foundation
- [x] `vtl/examples/vtl_vandermont/README.md` — VTL/Compute integration
- [x] `vtl/examples/vtl_opencl_vcl_support/README.md` — VCL/OpenCL support
- [x] Issue [#58](https://github.com/vlang/vtl/issues/58) — Phase 1: Vulkan Compute Foundation

### Phase 2 — Neural Network Forward Pass on GPU
- [x] Issue [#59](https://github.com/vlang/vtl/issues/59) — Phase 2: Forward Pass on Vulkan GPU

### Phase 3 — VSL Integration + CUDA Backend
- [x] Issue [#60](https://github.com/vlang/vtl/issues/60) — Phase 3: VSL + CUDA Backend
- [x] VTL imports from VSL: `import vtl.la` (LA operations), `import vtl.storage` (persistence)
- [x] CIFAR-10 dataset loader (`datasets/cifar10.v`)
- [x] `nn_cifar10` example with CUDA-aware config

### Phase 4 — GPU Autograd
- [x] Issue [#61](https://github.com/vlang/vtl/issues/61) — Phase 4: GPU Autograd
- [x] Autograd gates compile and run on CUDA context

### Phase 5 — OpenCL Backend
- [x] Issue [#62](https://github.com/vlang/vtl/issues/62) — Phase 5: OpenCL Backend (VCL unified into Compute abstraction)

### Phase 6 — ARM GPU Support
- [x] Issue [#63](https://github.com/vlang/vtl/issues/63) — Phase 6: ARM GPU Support (Android, iOS, Embedded)

### Phase 7+ — Performance Engineering
- [x] Issue [#64](https://github.com/vlang/vtl/issues/64) — Phase 7+: Kernel Fusion, Mixed Precision, Computation Graph Optimization

---

## 🎯 Active Development

### Phase A — CIFAR-10 End-to-End
- [x] CIFAR-10 path bug fixed (`datasets/cifar10.v`)
- [x] Configurable subset loading (`Cifar10Config.train_count`, `test_count`)
- [x] `nn_cifar10_safe` variant — safe defaults for dev/CI
- [x] `nn_cifar10_tiny` variant — real data subset (64 train, 16 test)
- [x] `nn_cifar10_tiny_synth` variant — synthetic data, no I/O (pipeline validation)
- [ ] **Open Issue**: Full `nn_cifar10` crashes on CI runner (OOM at compile,
  ~9 GB V compiler memory); runs on a local machine with enough RAM

### Phase B — DataLoader Infrastructure
- [x] `datasets/dataloader.v` — `DataLoader[T]` with batching, shuffle, labels ([issue #86](https://github.com/vlang/vtl/issues/86) closed)
- [x] Used in `nn_cifar10_*` examples

### Phase C — Model Serialization Polish — closed [#87](https://github.com/vlang/vtl/issues/87)
- [x] `serialization_test.v` round-trip at 1e-9
- [x] `nn_cifar10/main.v` checkpoints + `--resume` (PR #95)

### Phase C2 — VTL ↔ VSL GPU Integration — closed [#89](https://github.com/vlang/vtl/issues/89)–[#91](https://github.com/vlang/vtl/issues/91) (PR #93)
- [x] CUDA Linear, Conv2D, DeviceSession (opt-in `VTL_USE_CUDA=1`, `-d cuda`)

### Phase D — Numpy Benchmark Suite — closed [#88](https://github.com/vlang/vtl/issues/88) (PR #94)
- [x] `benchmarks/vs_numpy/` matmul + conv2d + `bench_util`
- [x] `autograd_bench.v` + `pytorch_baseline.py`
- [x] PR comment workflow (`benchmark-pr-comment.yml`, PR #96)

### Phase E — Example Consolidation
- [x] `nn_cifar10_cuda` — opt-in CUDA smoke (`VTL_USE_CUDA=1`, `-d cuda`)
- [x] `nn_cifar10_vulkan` — f32 smoke + `linear_forward_vulkan` check
- [x] Root `README.md` passes `v check-md -hide-warnings`
- [x] All example READMEs pass `v check-md -hide-warnings`

### Phase F — GPU training loop (follow-up to #91)
- [x] Phase 1: `DeviceSession` buffer reuse; CUDA forward for Linear/Conv2D
- [ ] Phase 2: GPU-resident `Variable` (forward tensors)
- [ ] Phase 3: CUDA backward for Linear / Conv2D
- [ ] Phase 4: Optimizer state on device

See [docs/DEVICE_MEMORY.md](docs/DEVICE_MEMORY.md).

---

## 🔜 Next priorities

| Priority | Work item |
|----------|-----------|
| P1 | [#41](https://github.com/vlang/vtl/issues/41) Windows crash |
| P1 | CI strategy for full `nn_cifar10` (compile memory) |
| P1 | Phases 2–4 in `DEVICE_MEMORY.md` |
| P2 | Vulkan wired into training (not only smoke example) |
| P2 | [#63](https://github.com/vlang/vtl/issues/63) ARM GPU |

**Project board:** [vlang org project #8](https://github.com/orgs/vlang/projects/8)

---

## 🧩 Open issues (VTL)

| # | Title | Priority | Notes |
|---|-------|----------|-------|
| [#41](https://github.com/vlang/vtl/issues/41) | Windows example crash | 🔴 P1 | |
| [#63](https://github.com/vlang/vtl/issues/63) | ARM GPU support | 🟡 P2 | Open |
| [#43](https://github.com/vlang/vtl/issues/43) | `stats.to_array` performance | 🟡 Medium | |
| [#40](https://github.com/vlang/vtl/issues/40) | YOLO for autograd gates | 🟡 Medium | |
| [#52](https://github.com/vlang/vtl/issues/52) | Tracel-AI/Burn reference | Research | |

**Closed ML epics:** #58–#64, #86–#91 — see [ML_ROADMAP.md](docs/ML_ROADMAP.md).

---

## 🔗 Cross-Repository Dependencies

| Dependency | Repo | Status |
|------------|------|--------|
| VSL LA ops (matmul, conv2d, solve, etc.) | [vlang/vsl](https://github.com/vlang/vsl) | ✅ Done |
| VSL CUDA (`vsl/cuda`, cuBLAS/cuDNN) | [vlang/vsl](https://github.com/vlang/vsl) | ✅ #280 |
| VSL Vulkan compute | [vlang/vsl](https://github.com/vlang/vsl) | ✅ #283–#284 |
| VSL OpenCL (VCL) | [vlang/vsl](https://github.com/vlang/vsl) | ✅ Integrated |
| vs NumPy / PR benchmarks | [vlang/vtl](https://github.com/vlang/vtl) | ✅ #88, workflow |

---

## Benchmark Strategy

### NumPy Baseline
All benchmarks compare VTL against NumPy using equivalent operations:

```python
# NumPy reference
import numpy as np
a = np.random.rand(1024, 1024)
b = np.random.rand(1024, 1024)
%timeit np.dot(a, b)
```

### VTL Implementation
```v
import vtl
import vtl.la

a := vtl.random[f64](0.0, 1.0, [1024, 1024], vtl.TensorData{})
b := vtl.random[f64](0.0, 1.0, [1024, 1024], vtl.TensorData{})
_ = la.matmul(a, b)!
```

### CI Integration
- GH Actions workflow runs `benchmarks/*.v` on every PR
- Posts comment with: VTL time, NumPy time, speedup ratio
- Fails PR if VTL is >3x slower than NumPy for equivalent ops (with justification for known gaps)

### Priority Benchmarks
1. **Matmul** — `f64[1024, 1024]` · CPU and CUDA
2. **Conv2D** — `f64[64, 3, 32, 32]` kernel `f64[3, 3, 3, 64]` · CPU and CUDA
3. **Autograd backprop** — MLP 3-layer, 256 hidden units
4. **Training step** — full forward + loss + backprop + optimizer update

---

## 📁 File Inventory

### VTL Examples
```
vtl/examples/
├── nn_cifar10/           # Full CIFAR-10 CNN (local machine recommended)
├── nn_cifar10_safe/      # Safe defaults, CI-friendly
├── nn_cifar10_tiny/      # Real data subset (64 train, 16 test)
├── nn_cifar10_tiny_synth/# Synthetic data, no I/O
├── nn_cifar10_cuda/      # CUDA smoke (opt-in)
├── nn_cifar10_vulkan/    # Vulkan GEMM smoke (opt-in)
├── nn_mnist/             # MNIST example
├── nn_xor/               # XOR training example
├── nn_regression_sine/   # Sine regression
├── nn_multiclass_iris/   # Iris multi-class
├── nn_simple_two_layer/  # Two-layer MLP
├── nn_autoencoder_simple/# Autoencoder
├── autograd_backprop/    # Backprop demonstration
├── vtl_basic_usage/      # Basic tensor operations
├── vtl_vandermont/       # Vandermonde matrix example
└── vtl_opencl_vcl_support/ # OpenCL/VCL support demo
```

### VTL Core
```
vtl/
├── nn/
│   ├── layers/           # Linear, Conv2D, MaxPool, LSTM, LayerNorm, Attention
│   ├── models/           # Sequential, Serialization
│   ├── optimizers/       # Adam, AdamW, RMSProp, AdaGrad, SGD
│   └── losses/           # MSE, BCE, CrossEntropy, Huber
├── datasets/
│   └── cifar10.v         # CIFAR-10 loader with subset support
├── autograd/
│   └── gates/            # Autograd gate implementations
├── benchmarks/vs_numpy/  # matmul, conv2d, autograd_bench
└── docs/                 # Tutorials + ML_ROADMAP
```

---

*Last updated: 2026-05-31* · Board: [project #8](https://github.com/orgs/vlang/projects/8)