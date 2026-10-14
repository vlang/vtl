# PyTorch Feature Parity

VTL aims to become a strong V-native alternative for tensor computing and
machine learning. This tracker compares VTL's documented public surface with
the major capability groups in the [PyTorch API reference](https://docs.pytorch.org/docs/stable/pytorch-api.html).
It is a work list, not a claim that VTL currently matches PyTorch's breadth,
performance, or ecosystem.

## Open implementation issues

| Capability | Issue |
|------------|-------|
| OpenCL compute and neural-network dispatch | [#750](https://github.com/vlang/vtl/issues/750) |
| Vulkan activation residency across sequential layers | [#797](https://github.com/vlang/vtl/issues/797) |
| CUDA backward for remaining neural-network layers | [#161](https://github.com/vlang/vtl/issues/161) |
| Tensor semantics; dtype and device model | [#731](https://github.com/vlang/vtl/issues/731), [#732](https://github.com/vlang/vtl/issues/732) |
| Higher-order autograd | [#733](https://github.com/vlang/vtl/issues/733) |
| Functional tensor transforms such as `vmap` | [#784](https://github.com/vlang/vtl/issues/784) |
| Core layer-family coverage | [#802](https://github.com/vlang/vtl/issues/802) |
| Loss function coverage | [#803](https://github.com/vlang/vtl/issues/803) |
| Stacked recurrent APIs; custom module state | [#734](https://github.com/vlang/vtl/issues/734), [#735](https://github.com/vlang/vtl/issues/735) |
| Optimizer/scheduler parity | [#736](https://github.com/vlang/vtl/issues/736) |
| Device-resident CUDA optimizer state and updates | [#758](https://github.com/vlang/vtl/issues/758) |
| INT8 quantization and inference kernels | [#792](https://github.com/vlang/vtl/issues/792) |
| Data transforms and scalable loading | [#737](https://github.com/vlang/vtl/issues/737), [#738](https://github.com/vlang/vtl/issues/738) |
| Checkpoint migration and safe loading | [#739](https://github.com/vlang/vtl/issues/739) |
| Graph execution and memory profiling | [#740](https://github.com/vlang/vtl/issues/740), [#741](https://github.com/vlang/vtl/issues/741) |
| Distributed training and mixed precision | [#742](https://github.com/vlang/vtl/issues/742) |
| Interoperability and deployment | [#743](https://github.com/vlang/vtl/issues/743) |
| Sparse tensor formats and operations | [#746](https://github.com/vlang/vtl/issues/746) |
| In-process tensor buffer interoperability | [#748](https://github.com/vlang/vtl/issues/748) |
| PyTorch migration and training tutorials | [#770](https://github.com/vlang/vtl/issues/770) |
| Matched VTL/PyTorch performance benchmarks | [#771](https://github.com/vlang/vtl/issues/771) |

## Capability matrix

### Tensor operations

**VTL today:** Generic tensors, broadcasting, indexing, reductions, statistics,
FFT, and VSL-backed linear algebra. See the [NumPy parity tracker](./NUMPY_PARITY.md).

`Tensor.masked_scatter` and `Variable.masked_scatter` support broadcastable
masks, sequential source consumption, and autograd for selected source values.
`vtl.roll` matches circular flattened and axis-wise shifts;
`Variable.roll` and `Variable.roll_shifts` propagate gradients with inverse
shifts. `Variable.rot90`, `Variable.rot90_k`, and `Variable.rot90_axes` rotate
tensor planes and propagate gradients through the inverse rotation.
`Variable.flip` propagates gradients through the self-inverse axis reversal.
`Variable.repeat` and `Variable.repeat_axis` sum gradients across copies.
`Variable.tile` reduces gradients from each repeated block to the input.
`Variable.pad` routes gradients according to its constant, edge, wrap, reflect,
or symmetric boundary mode.

**Next:** Audit operator coverage and semantics against `torch.Tensor`, including
dtype promotion, scalar behavior, strides, views, in-place operations, and error
cases.

### Dtypes and devices

**VTL today:** V supports multiple numeric types; optional CUDA and Vulkan paths
exist for selected operations. OpenCL currently covers tensor transfer and a
small `add_scalar` smoke path; broader compute and neural-network dispatch are
tracked in [#750](https://github.com/vlang/vtl/issues/750).

**Next:** Define a consistent dtype promotion and device model. Broaden backend
coverage and test transfers, mixed precision, and device-specific behavior.

VTL now has an opt-in `vtl.quantization` foundation for signed INT8 affine
per-tensor and per-channel conversion, min/max calibration observers, and
f32/f64 dequantization. Quantized Linear/Conv kernels, layer conversion, and
accelerator execution are still open in [#792](https://github.com/vlang/vtl/issues/792);
the current API makes no inference performance claim.

### Autograd

**VTL today:** Reverse-mode differentiation for supported tensor operations,
including cumulative sum and product, selected neural-network layers, and
differentiable multi-axis sum and mean reductions (`Variable.sum_along_axes`
and `Variable.mean_along_axes`; implemented in [#753](https://github.com/vlang/vtl/issues/753)).
Floating-point `Variable.remainder` also propagates gradients to both operands,
reduces broadcast gradients to their input shapes, and follows PyTorch's
`floor(a / b)` divisor-gradient convention at exact multiples.

**Next:** Expand operation coverage and add gradient APIs and graph controls
comparable to `torch.autograd`, including higher-order gradients and retained
graphs where feasible.

### Neural-network modules

**VTL today:** `Sequential`, common activations, linear, convolutional,
recurrent, attention, and normalization layers. Typed CPU LSTM layers support
stacked hidden/cell state through `forward_with_state`; typed CPU GRU layers
support an explicit initial and final hidden state. Their default input layouts
differ: LSTM is `[batch, sequence, features]`, while GRU is
`[sequence, batch, features]`. The typed GRU API also provides
`forward_batch_first` and `forward_batch_first_with_state` for
`[batch, sequence, features]` input; `Sequential.gru` keeps the sequence-first
layout. See the
[NN tutorial](./TUTORIAL_NEURAL_NETWORKS.md).

GroupNorm supports channel-first `[batch, channels, ...]` inputs, per-channel
affine parameters, and CPU input/parameter gradients. RMSNorm normalizes trailing
dimensions by their root mean square and supports optional per-element weights.
Both are available as layers and in `Sequential`, with CPU autograd; device
kernels and broader module parity remain separate work.

`BatchNorm2D` normalizes `[batch, channels, height, width]` inputs per channel
with train/eval modes, running statistics (the running variance uses the
unbiased batch variance, as in PyTorch), optional affine parameters, and CPU
autograd for input, gamma and beta. It is available as a layer and in
`Sequential` via `batchnorm2d`, closing the BatchNorm2d gap next to the
existing `BatchNorm1D`. `BatchNorm3D` provides the same contract for
`[batch, channels, depth, height, width]` inputs via `batchnorm3d`.
Device kernels for batch normalization remain separate work.

`Sequential.gru` can be called repeatedly to compose a stack of single-layer
GRUs, each with a zero initial hidden state. This composes stacked forward
passes but does not expose PyTorch's single-module `num_layers` or stacked
hidden-state contract.

**Next:** Add bidirectional recurrent layers, a PyTorch-style stacked recurrent
state API, recurrent dropout and variable-length packed sequences. `Sequential`
supports `train()`/`eval()` behavior for Dropout, BatchNorm1D, BatchNorm2D and
BatchNorm3D (mode propagation and running-statistics behavior are covered by
`nn/models/sequential_normalization_test.v`); LayerNorm, GroupNorm and RMSNorm
have no training-dependent statistics, matching PyTorch. Extending mode-aware
behavior to custom modules remains open. Parameter and
buffer registration and composable user-defined modules remain gaps. Stateful
LSTM/GRU entry points are available today, but do not imply parity with the
full `torch.nn` recurrent API or CUDA backward.

### Losses and optimizers

**VTL today:** Common classification and regression losses and optimizers, with
CPU training and experimental GPU paths. Learning-rate schedulers include
`StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`,
`CosineAnnealingWarmRestarts`, `LinearLR`, `PolynomialLR`, `ConstantLR`,
`CyclicLR`, `OneCycleLR`, and
`ReduceLROnPlateau`. `OneCycleLR` implements the learning-rate curve (cosine or
linear, two or three phases) and inverse momentum cycling with configurable
bounds. `CyclicLR` includes
the triangular, triangular2, and exp_range policies and computes the inverse
momentum curve with configurable bounds. Unlike PyTorch, VTL returns that
momentum value for the caller to apply; it does not mutate optimizer state.
`CyclicLR` accepts a custom scale callback that receives the cycle number or
minibatch step, selected by `scale_mode`; cycle callbacks use the same one-based
cycle count as PyTorch.
`SgdOptimizer` supports coupled weight decay, momentum, dampening, and Nesterov
updates, including PyTorch's first-step momentum-buffer initialization.
Gradient clipping includes global L2 clipping and a configurable global
p-norm entry point (including zero-count and positive-infinity norms), which
return the pre-clipping norm, plus element-wise value clipping with a
non-negative limit. Both update
trainable gradients in place; norm clipping supports f32/f64 gradients and
rejects non-finite values. Optimizer integration and PyTorch's foreach/device
behavior remain gaps.
`get_total_grad_norm` computes the norm without mutation, and
`clip_grads_with_norm` applies a previously computed norm, matching the split
offered by PyTorch's `get_total_norm` and `clip_grads_with_norm_` APIs.
`ReduceLROnPlateau.step(current_lr, metric)` tracks the best metric and supports
PyTorch-style min/max modes, relative/absolute thresholds, patience, cooldown,
minimum LR, and epsilon. It returns the next LR for the caller to apply.
All seven optimizers (`SgdOptimizer`, `AdamOptimizer`, `AdamWOptimizer`,
`RMSPropOptimizer`, `AdaGradOptimizer`, `NAdamOptimizer` and
`RAdamOptimizer`) now support PyTorch-style parameter groups through
`add_param_group`, each with its own learning rate and optimizer-specific
settings and independent state buffers (including per-group step counters
and NAdam momentum products); `set_group_learning_rate` is the scheduler
integration point.

`nll_loss` supports weighted class-index and one-hot/soft targets, mean and sum
reductions, per-position `none` reduction, and `ignore_index` for class-index
targets. Weighted class-index means divide by the sum of non-ignored target
weights; soft-target means divide by the number of samples.
`nll_loss_indices` accepts `Tensor[int]` targets; the regular generic loss also
accepts same-type integer-valued target tensors. Both APIs accept unbatched
`[C]` log-probabilities with a scalar class-index target and batched
`[N, C, ...]` inputs with spatial dimensions after the class axis. Unbatched
`reduction: 'none'` returns a scalar, and all reductions propagate gradients.

`cross_entropy_loss_with_options` supports optional class weights,
`ignore_index`, and `none`, `mean`, or `sum` reductions for unbatched `[C]` or
batched/spatial `[N, C, ...]` logits. `cross_entropy_loss_indices` accepts
`Tensor[int]` targets; the generic loss accepts same-type integer-valued targets
or matching one-hot/soft targets. Weighted class-index means divide by the sum
of non-ignored target weights; soft-target means divide by the number of samples.
`label_smoothing` mixes targets with the uniform class distribution. All paths
propagate gradients.

**Next:** Audit formulas, defaults, reduction modes, sparse gradients, and
numerical behavior against `torch.nn` and `torch.optim`. Parameter groups are
implemented for all current optimizers. Scheduler integration and remaining
backend differences are tracked in [#736](https://github.com/vlang/vtl/issues/736).

### Data pipeline

**VTL today:** A tensor-backed `DataLoader` batches rows and optional labels,
supports seeded shuffling, `drop_last`, and sequential, random, or subset-random
sample ordering through `new_with_sampler`. `WeightedRandomSampler` supports
weighted draws with or without replacement, and the tensor-backed
`vtl.datasets` API ships MNIST, IMDB, and CIFAR-10 loaders. Its defaults are
`batch_size: 32`, `shuffle: true`, and
`drop_last: true`; PyTorch's `DataLoader` defaults are `1`, `false`, and
`false`, respectively. `FixedBatchSampler` supports custom batch boundaries
and `drop_last`. `DataLoader.set_collate_fn`, `collated_batch`, and
`for_each_collated` support custom tensor collation from the selected source
rows and optional labels.
The dense tensor API returns feature and label tensors; it does not represent
arbitrary Python objects or nested sample structures. Iterable datasets,
worker processes, and pinned-memory loading remain unavailable.

**Next:** Add composable transforms, worker and process loading, streaming
datasets, generator injection, and stronger reproducibility controls. See the
[custom collation example](../examples/datasets_custom_collate/main.v) and the
[PyTorch data loading API](https://docs.pytorch.org/docs/stable/data.html)
when porting a pipeline; specify VTL loader options explicitly because its
defaults differ.

### Serialization

**VTL today:** NumPy `.npy`/`.npz` interchange and neural-network checkpoint
support. The versioned JSON checkpoint API saves model weights and training
metadata; Adam checkpoint helpers preserve configuration and moment buffers,
AdamW helpers preserve configuration and moment buffers, and SGD helpers
preserve configuration, momentum buffers, and first-step momentum
initialization state. RMSProp helpers preserve configuration and
squared-gradient buffers. AdaGrad helpers preserve configuration and accumulated
squared gradients. NAdam checkpoints preserve the scheduled momentum product;
RAdam checkpoints preserve the rectification step. Both persist configuration
and first/second moments.

**Next:** Add device remapping, version migration, and safe loading behavior.
PyTorch pickle compatibility is not implied.

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
- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Optimizers](https://docs.pytorch.org/docs/stable/optim.html)
- [Data loading](https://docs.pytorch.org/docs/stable/data.html)
- [Serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html)
- [Linear algebra](https://docs.pytorch.org/docs/stable/linalg.html)
- [FFT](https://docs.pytorch.org/docs/stable/fft.html)
