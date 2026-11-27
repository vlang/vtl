# `vtl.nn.types`

Shared adapter types for neural-network components. `Layer[T]` erases a
concrete layer behind callbacks for output shape, variables, and forward
execution; `Loss[T]` provides a common loss interface; `Optimizer[T]` exposes
registered variables, learning rate, parameter building, and update. These
adapters let the sequential model and optimizers work with multiple
implementations.

This package is mainly useful when implementing custom layers or integrating
new components. Most application code can use constructors from
[`layers`](../layers/README.md), [`loss`](../loss/README.md), and
[`models`](../models/README.md).

The contracts are defined in [`layer.v`](layer.v), [`loss.v`](loss.v), and
[`optimizer.v`](optimizer.v). Implementations should keep tensor shapes and
context ownership consistent across forward and backward calls.

`Module[T]` is a state registry for custom components. Register trainable
`Variable[T]` values with `register_parameter`, non-trainable tensor state with
`register_buffer`, and nested registries with `register_module`. Names must be
non-empty, cannot contain dots, and must be unique within each module across
parameters, buffers, and children. `named_parameters` and `named_buffers`
traverse nested registries depth-first using dotted names; `parameters` and
`buffers` return their values without names. Cyclic child graphs are rejected.

This registry does not yet provide custom forward/mode hooks or automatically
adapt state to `Sequential`, optimizer construction, or checkpoint
serialization. Existing `Layer[T]` wrappers and `Sequential` behavior are
unchanged.
