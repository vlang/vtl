# `vtl.storage`

Storage backends used by VTL tensors. CPU storage is the default general-purpose
path; optional CUDA, Vulkan, and VCL implementations are selected through V
compile-time defines and are not interchangeable in availability or maturity.

Most applications should use `vtl.Tensor[T]` and its creation/manipulation
functions instead of constructing storage directly. The backend modules expose
lower-level allocation and access primitives for library and backend work.

CPU storage constructors from arrays copy their input; `clone` creates an
independent buffer, while `offset` refers to a range in the existing storage.
At tensor level, `view`, `slice`, and stride-only transforms can share backing
storage when no layout conversion is needed; `copy` explicitly allocates
independent data. Treat mutations through a view as mutations of shared data,
and check whether an operation materializes a new layout before relying on
zero-copy behavior. This is a CPU tensor contract, not a promise that each
accelerator backend implements identical aliasing.

CPU is the default backend. CUDA and VCL expose backend-specific storage behind
compile flags; Vulkan has an implemented conditional backend and a no-backend
fallback. These accelerator paths have different allocation, synchronization,
transfer, and lifetime constraints and remain experimental. See
[device memory notes](../docs/DEVICE_MEMORY.md) before using them.
