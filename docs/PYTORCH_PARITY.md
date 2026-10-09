# PyTorch Feature Parity

VTL aims to become a strong V-native alternative for tensor computing and
machine learning. This tracker compares VTL's documented public surface with
the major capability groups in the [PyTorch API reference](https://docs.pytorch.org/docs/stable/pytorch-api.html).
It is a work list, not a claim that VTL currently matches PyTorch's breadth,
performance, or ecosystem.

## Capability matrix

### Tensor operations

**VTL today:** Generic tensors, broadcasting, indexing, reductions, statistics,
FFT, and VSL-backed linear algebra. See the [NumPy parity tracker](./NUMPY_PARITY.md).

**Next:** Audit operator coverage and semantics against `torch.Tensor`, including
dtype promotion, scalar behavior, strides, views, in-place operations, and error
cases.

### Dtypes and devices

**VTL today:** V supports multiple numeric types; optional CUDA and Vulkan paths
exist for selected operations.

**Next:** Define a consistent dtype promotion and device model. Broaden backend
coverage and test transfers, mixed precision, and device-specific behavior.

### Autograd

**VTL today:** Reverse-mode differentiation for supported tensor operations and
selected neural-network layers.

**Next:** Expand operation coverage and add gradient APIs and graph controls
comparable to `torch.autograd`, including higher-order gradients and retained
graphs where feasible.

### Neural-network modules

**VTL today:** `Sequential`, common activations, linear, convolutional,
recurrent, attention, and normalization layers. See the
[NN tutorial](./TUTORIAL_NEURAL_NETWORKS.md).

**Next:** Expand module and configuration coverage, stateful recurrent
interfaces, initialization options, train/eval behavior, parameter and buffer
registration, and composable user-defined modules.

### Losses and optimizers

**VTL today:** Common classification and regression losses and optimizers, with
CPU training and experimental GPU paths.

**Next:** Audit formulas, defaults, reduction modes, parameter groups, scheduler
support, sparse gradients, and numerical behavior against `torch.nn` and
`torch.optim`.

### Data pipeline

**VTL today:** Dataset and DataLoader APIs, plus MNIST, IMDB, and CIFAR-10
loaders.

**Next:** Add composable transforms, samplers, batching and collation controls,
worker and process loading, streaming datasets, and stronger reproducibility
controls.

### Serialization

**VTL today:** NumPy `.npy`/`.npz` interchange and neural-network checkpoint
support.

**Next:** Define a stable model and optimizer checkpoint format, device
remapping, version compatibility, and safe loading behavior. PyTorch pickle
compatibility is not implied.

### Compilation and execution

**VTL today:** Eager V execution with optional backend kernels.

**Next:** Add graph capture or compilation, operator fusion, memory planning,
and profiling tools before comparing with `torch.compile` or graph execution.

### Distributed and mixed precision

**VTL today:** VSL includes selected backend and MPI support; VTL's training
surface is primarily local.

**Next:** Establish tensor and gradient synchronization, distributed optimizers,
autocast and scaling, and multi-device training contracts with dedicated tests.

### Ecosystem

**VTL today:** V-native documentation, tutorials, examples, and VSL integration.

**Next:** Grow interoperability, pretrained model availability, deployment and
export paths, community examples, and migration guidance.

## Verification rules

- Mark a capability complete only when its public API, tests, documentation,
  and a runnable example cover the promised behavior.
- Record semantic differences explicitly. Similar names do not imply identical
  defaults, broadcasting, dtype promotion, gradients, or error behavior.
- Treat CPU, CUDA, Vulkan, OpenCL, and MPI as separate validation targets. A
  feature implemented for one backend is not evidence that it works on all.
- Compare performance with reproducible workloads, equivalent precision,
  backend, thread count, warm-up, and memory limits. A single kernel benchmark
  does not establish framework-wide performance parity.
- Link each completed row or capability to the implementation, tests, and
  tutorial that support it. Keep unsupported PyTorch contracts clearly listed
  until they are implemented and validated.

## Primary references

- [PyTorch API reference](https://docs.pytorch.org/docs/stable/pytorch-api.html)
- [Tensor construction](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html)
- [Autograd](https://docs.pytorch.org/docs/stable/autograd.html)
- [Neural-network modules](https://docs.pytorch.org/docs/stable/nn.html)
- [Optimizers](https://docs.pytorch.org/docs/stable/optim.html)
- [Data loading](https://docs.pytorch.org/docs/stable/data.html)
- [Serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
- [Linear algebra](https://docs.pytorch.org/docs/stable/linalg.html)
- [FFT](https://docs.pytorch.org/docs/stable/fft.html)
