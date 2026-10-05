# `vtl.storage`

Storage backends used by VTL tensors. CPU storage is the default general-purpose
path; optional CUDA, Vulkan, and VCL implementations are selected through V
compile-time defines and are not interchangeable in availability or maturity.

Most applications should use `vtl.Tensor[T]` and its creation/manipulation
functions instead of constructing storage directly. The backend modules expose
lower-level allocation and access primitives for library and backend work.
Storage owns the element buffer; device transfers, synchronization, and lifetime
rules are backend-specific. See [device memory notes](../docs/DEVICE_MEMORY.md)
before using accelerator storage.
