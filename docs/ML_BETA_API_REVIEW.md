# VTL ML Beta Public API Review

This review captures the public API state for the ML beta milestone. It focuses
on user-facing tensor, autograd, neural-network, optimizer, dataset, storage, and
GPU APIs.

## Documentation Coverage

The repository now includes `tools/audit_public_docs.py`, which checks that each
public declaration has an immediate `//` comment on the previous line.

Current targeted result:

```sh
python3 tools/audit_public_docs.py --summary
# missing_public_docs=0
```

The audit covers public `fn`, `struct`, `interface`, `enum`, `type`, and `const`
declarations in tracked V files. Test files, generated shader blobs, C shims,
examples, and benchmarks are treated as non-release API by default.

## Beta-Stable Surface

These modules form the beta-facing API and should remain source-compatible
through the beta unless a breaking change is explicitly approved:

- `vtl`: tensor creation, shape/layout helpers, indexing, broadcasting,
  reductions, math operations, random creation, casting, stacking, and splitting.
- `vtl.autograd`: contexts, variables, payloads, gates, and variable operations.
- `vtl.nn.models`: `Sequential` construction, layer composition, forward pass,
  and configured loss usage.
- `vtl.nn.layers`: Linear, Conv2D, activations, pooling, normalization,
  embedding, recurrent, and attention layer factories/configuration structs.
- `vtl.nn.loss`: MSE, BCE, cross-entropy, sigmoid/softmax cross-entropy, Huber,
  KL divergence, and NLL helpers.
- `vtl.nn.optimizers`: Adam, AdamW, SGD, RMSProp, AdaGrad, schedulers, and
  CPU optimizer steps used by the f32 training path.
- `vtl.datasets` and `vtl.nn.data`: loaders and dataloader utilities.
- `vtl.storage`: CPU and optional backend storage abstractions used by tensors.

## Experimental Public Surface

The following public APIs should be documented and treated as experimental
rather than stable end-user contracts:

- CUDA/Vulkan hook functions in `nn/layers/*cuda*`, `nn/layers/*vulkan*`,
  `nn/gates/**`, `autograd_cuda/**`, `tensor_cuda_*`, and `tensor_vulkan_*`.
- CUDA/Vulkan optimizer step helpers, including low-level Adam GPU parity paths.
- Public fallback functions with names like `try_*`, `*_hooks_*`, `*_bind_*`, and
  `*_notd_*`.
- Low-level storage constructors in `storage/cuda_*`, `storage/vcl_*`, and
  `storage/vulkan_*`.

These APIs are useful for conditional compilation and backend dispatch, but they
should not be advertised as the primary user path. User docs should prefer
high-level tensor/model APIs and environment/config flags.

## API Coherence Findings

### Should Keep

- Factory naming is mostly coherent: `*_layer`, `*_optimizer`, `sequential_*`,
  and config structs such as `Conv2DConfig` and `AdamOptimizerConfig`.
- The `Sequential` API gives a compact learning path for users coming from other
  ML frameworks while still exposing lower-level layers.
- Tensor functions are broadly NumPy-like and discoverable: `zeros`, `ones`,
  `full`, `eye`, `range`, `seq`, `from_array`, `to_array`, reductions, and
  broadcasting helpers.

### Should Clarify Before Beta

- `Tensor` exposes mutable fields (`data`, `memory`, `size`, `shape`, `strides`).
  That is convenient for current internals but makes invariants easy to break.
  If this cannot change before beta, docs should call these fields low-level.
- `Tensor.copy(memory)` and `Tensor.view()` should remain documented around
  ownership: `copy` owns cloned storage; `view` shares storage.
- GPU methods on tensors should state when they copy data, when they return the
  same tensor, and which build flags enable them.
- `nn/internal/**` still has public functions. If V currently requires this for
  package boundaries, document them as internal support APIs and avoid tutorial
  references.
- Optimizer and training examples should prefer `f32` for the beta path.
  `f64` tensor math and autograd/training now have mixed `f32`/`f64` regression
  coverage, but `f32` remains the primary beta training path advertised in user
  docs.
- Custom `Gate`, `Layer`, and `Loss` extension points use opaque dispatch
  wrappers internally to avoid V generic interface specialization drift. The
  high-level user APIs remain stable, while third-party custom extension
  implementations should be treated as experimental during beta.

### Breaking-Change Candidates

Do not change these without explicit approval:

- Make `Tensor` fields private and expose accessors for shape, strides, memory,
  and storage.
- Move `nn/internal/**` public helpers behind a private/internal boundary.
- Rename backend hooks (`try_*`, `*_hooks_*`, `*_bind_*`) to a consistent
  experimental namespace.
- Normalize CUDA and Vulkan optimizer entry points under one backend dispatch
  abstraction.

## Beta Recommendation

VTL is suitable for an ML beta if the public contract is described as:

1. Stable: tensors, autograd, high-level layers/losses/optimizers/models,
   datasets, and f32 CPU training.
2. Beta/experimental: CUDA and Vulkan acceleration hooks, low-level storage, and
   conditional backend dispatch.
3. Internal support: `nn/internal/**` and backend-specific glue that remains
   public for compilation reasons.

Windows, ARM GPU, persistent Vulkan activation chaining, and backend-specific
performance work are tracked as validation or post-beta work. They should not be
advertised as part of the stable beta contract until dedicated CI or hardware
coverage proves them.

