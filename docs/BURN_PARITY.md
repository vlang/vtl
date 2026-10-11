# Burn Capability Comparison

This comparison addresses [VTL issue #52][issue52]. It is a roadmap aid, not a
performance result or a claim that VTL and Burn have equivalent APIs.

**Upstream snapshot:** Burn `main` README and linked backend, autodiff, and
fusion documentation reviewed on 2026-10-11. Recheck these sources before
using this as a current product comparison.

## Capability comparison

### Tensor API

- **Burn:** A unified tensor API with dynamic shapes and backend selection.
- **VTL:** Generic tensors, broadcasting, indexing, reductions, FFT, and linear
  algebra; see [NumPy parity](./NUMPY_PARITY.md).
- **Gap:** Continue the NumPy semantic audit for dtype promotion, views,
  strides, indexing, and error behavior.

### CPU and accelerator backends

- **Burn:** CubeCL provides CPU, CUDA, ROCm, Metal, Vulkan, and WebGPU. Flex is
  a separate CPU backend for ARM, Wasm, and `no_std`.
- **VTL:** Pure-V CPU plus optional CBLAS/LAPACKE and selected CUDA, Vulkan,
  and OpenCL/VCL paths through VSL. See [device memory](./DEVICE_MEMORY.md)
  and [VSL/VTL CUDA notes](./VSL_VTL_CUDA.md).
- **Gap:** Publish backend support per operation. Metal, ROCm, and WebGPU are
  not documented VTL backends. ARM GPU support remains in [issue #63][issue63].

### Autodiff

- **Burn:** Autodiff decorates a concrete backend and can be enabled on a
  device before tensors and models are created.
- **VTL:** Reverse-mode autograd covers selected tensor operations and NN
  layers. CPU paths are broader than experimental GPU backward paths; see
  [PyTorch parity](./PYTORCH_PARITY.md).
- **Gap:** Expand operation and device coverage, document graph lifetime and
  gradient-retention semantics, and validate gradients across backends.

### Kernel fusion

- **Burn:** A Fusion backend decorator supports automatic fusion on accelerated
  backends that provide it.
- **VTL:** Specific fused paths exist, including the Vulkan Adam shader.
  General operation-chain fusion remains open in [issue #40][issue40].
- **Gap:** Grow fusion incrementally. Require numerical tests and end-to-end
  benchmarks for each fused workload.

### Training and models

- **Burn:** Provides model modules, a `Learner`, custom training loops,
  checkpointing, and curated model examples.
- **VTL:** Provides `Sequential`, layers, losses, optimizers, model
  serialization, datasets, and training examples.
- **Gap:** Add broader recurrent features, optimizer checkpointing, and
  end-to-end examples as tracked in [PyTorch parity](./PYTORCH_PARITY.md).

### Data

- **Burn:** The ecosystem includes `burn-dataset`, transforms, and ready-made
  dataset sources.
- **VTL:** Includes MNIST, IMDB, and CIFAR-10 loaders and a `DataLoader`.
- **Gap:** Add composable transforms, samplers, worker loading, streaming, and
  reproducibility controls.

### Interoperability

- **Burn:** Documents ONNX model import and `burn-store` support for PyTorch
  and Safetensors weights.
- **VTL:** Documents NumPy `.npy`/`.npz` interchange and VTL model checkpoints.
- **Gap:** Define a stable model interchange path with explicit compatibility
  tests, starting from a scoped ONNX or Safetensors proposal.

### Checkpointing and distributed execution

- **Burn:** Documents gradient checkpointing and a beta remote-execution
  backend decorator.
- **VTL:** Training is primarily local. VSL MPI support alone does not provide
  distributed VTL autograd or optimizer semantics.
- **Gap:** Design activation checkpointing and a VTL distributed-training
  contract before claiming parity.

### Examples and ecosystem

- **Burn:** Lists a broad example set, pretrained models, and related vision,
  RL, data, and benchmark projects.
- **VTL:** Provides tutorials, examples, and a feature catalog.
- **Gap:** Grow runnable task examples and model coverage; tie claims to
  artifacts that build and run in CI.

## Priorities for VTL

1. **Define backend contracts.** Track operation, dtype, device, backward
   support, and fallback separately. A VSL backend does not imply that every
   VTL operation uses it.
2. **Measure full workloads.** Compare VTL with PyTorch or Burn using the same
   model, dtype, device, thread count, warm-up, and optimizer work. Report
   transfers and setup separately from kernel-only timings.
3. **Build general fusion incrementally.** Preserve eager execution as the
   reference and compare outputs and gradients before reporting speedups.
4. **Close deployment gaps.** Prioritize stable checkpointing and an
   interoperable model format before expanding remote or distributed claims.

These priorities complement the narrower [NumPy parity](./NUMPY_PARITY.md) and
[PyTorch parity](./PYTORCH_PARITY.md) trackers. Burn is an architecture and
ecosystem reference; NumPy and PyTorch remain the direct compatibility and
benchmark references for their respective VTL domains.

## Sources

- [Burn repository README](https://github.com/tracel-ai/burn) — backend matrix,
  autodiff, fusion, remote execution, ecosystem, and examples.
- [VTL NumPy parity](./NUMPY_PARITY.md)
- [VTL PyTorch parity](./PYTORCH_PARITY.md)
- [VTL ML roadmap](./ML_ROADMAP.md)

[issue40]: https://github.com/vlang/vtl/issues/40
[issue52]: https://github.com/vlang/vtl/issues/52
[issue63]: https://github.com/vlang/vtl/issues/63
