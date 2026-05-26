# VTL Neural Network Roadmap

> **Goal**: Production-grade Neural Network support in VTL with CUDA/GPU acceleration, DataLoader, CIFAR-10 example, and model serialization.

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
- [ ] **Open Issue**: Full `nn_cifar10` example crashes on this runner (OOM at compile time, ~9 GB V compiler memory); runs correctly on user's local machine

### Phase B — DataLoader Infrastructure
- [ ] Implement `DataLoader[T]` struct with:
  - `batch_size`, `shuffle`, `drop_last` options
  - `next() ![]Tensor[f64]` method
  - `len() int` method
  - `reset()`
- [ ] Connect to existing `Cifar10Config` for seamless mini-batch training

### Phase C — Model Serialization Polish
- [ ] `nn/models/serialization.v` — `to_json()` / `from_json()` round-trip test
- [ ] Save/load CIFAR-10 model weights to disk
- [ ] Integrate serialization into `nn_cifar10` example as checkpointing

### Phase D — Numpy Benchmark Suite
- [ ] `vtl/benchmarks/` directory with:
  - `matmul_benchmark.v` — VTL vs NumPy: matrix multiplication (CPU + CUDA)
  - `conv2d_benchmark.v` — VTL vs NumPy: 2D convolution
  - `autograd_benchmark.v` — VTL vs PyTorch: backprop overhead
  - `nn_training_benchmark.v` — end-to-end training step (VTL vs PyTorch)
- [ ] CI benchmark runner (GH Actions) with comparison vs NumPy baseline
- [ ] Report speedup/slowdown metrics in PRs

### Phase E — Example Consolidation
- [ ] `nn_cifar10_cuda` variant with explicit CUDA device selection
- [ ] `nn_cifar10_vulkan` variant using Vulkan backend
- [ ] All variants should pass `v check-md -fix -hide-warnings` from `~/.vmodules`

---

## 🧩 Existing Open Issues (VTL)

| # | Title | Priority | Notes |
|---|-------|----------|-------|
| [#64](https://github.com/vlang/vtl/issues/64) | Phase 7+: Performance Engineering | 🔴 High | Kernel fusion, mixed precision |
| [#63](https://github.com/vlang/vtl/issues/63) | Phase 6: ARM GPU Support | 🟢 Done | |
| [#62](https://github.com/vlang/vtl/issues/62) | Phase 5: OpenCL Backend | 🟢 Done | |
| [#61](https://github.com/vlang/vtl/issues/61) | Phase 4: GPU Autograd | 🟢 Done | |
| [#60](https://github.com/vlang/vtl/issues/60) | Phase 3: VSL Integration + CUDA | 🟢 Done | |
| [#59](https://github.com/vlang/vtl/issues/59) | Phase 2: Forward Pass on GPU | 🟢 Done | |
| [#58](https://github.com/vlang/vtl/issues/58) | Phase 1: Vulkan Compute Foundation | 🟢 Done | |
| [#57](https://github.com/vlang/vtl/issues/57) | GPU Architecture: Multi-Backend | 🟢 Done | |
| [#52](https://github.com/vlang/vtl/issues/52) | Tracel-AI/Burn reference | 🟢 Done | Architecture aligned with Burn |
| [#43](https://github.com/vlang/vtl/issues/43) | `stats.to_array` performance fix | 🔴 High | Prevent allocation overhead |
| [#40](https://github.com/vlang/vtl/issues/40) | YOLO for autograd gates | 🟡 Medium | Loop-once pattern for gates |
| [#39](https://github.com/vlang/vtl/issues/39) | Remove continue-on-error in v fmt | 🔴 High | |
| [#30](https://github.com/vlang/vtl/issues/30) | Add `vsl.plot` examples | 🟡 Medium | |
| [#29](https://github.com/vlang/vtl/issues/29) | Improve VTL tutorial | 🟡 Medium | |
| [#28](https://github.com/vlang/vtl/issues/28) | Add ML metrics examples | 🟡 Medium | |
| [#27](https://github.com/vlang/vtl/issues/27) | Add more datasets | 🟡 Medium | CIFAR-10 done; MNIST, ImageNet pending |
| [#26](https://github.com/vlang/vtl/issues/26) | Add more examples | 🟡 Medium | |
| [#25](https://github.com/vlang/vtl/issues/25) | autograd:register inconsistency | 🟡 Medium | |
| [#23](https://github.com/vlang/vtl/issues/23) | datasets example compile error | 🔴 High | Fixed |

---

## 🔗 Cross-Repository Dependencies

| Dependency | Repo | Status |
|------------|------|--------|
| VSL LA ops (matmul, conv2d, solve, etc.) | [vlang/vsl](https://github.com/vlang/vsl) | ✅ Done |
| VSL CUDA backend (`vsl/cuda`) | [vlang/vsl](https://github.com/vlang/vsl) | ✅ Phase B done |
| VSL Vulkan backend (`vsl/vulkan`) | [vlang/vsl](https://github.com/vlang/vsl) | 🔴 Phase A in progress |
| VSL OpenCL backend (VCL) | [vlang/vsl](https://github.com/vlang/vsl) | 🟡 VCL integrated |
| Benchmark harness | [vlang/vtl](https://github.com/vlang/vtl) | 🔴 Not started |

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
// VTL benchmark
import vtl
a := vtl.rand[f64]([1024, 1024])
b := vtl.rand[f64]([1024, 1024])
sw := stats.new_stopwatch()
c := vtl.la.matmul(a, b)!
println('VTL matmul: ${sw.elapsed()}')
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
├── benchmarks/           # (TODO) VTL vs NumPy benchmarks
└── docs/                # Tutorials
```

---

*Last updated: 2026-05-26*