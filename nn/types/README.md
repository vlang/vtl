# `vtl.nn.types`

Shared adapter types for neural-network components. `Layer[T]` erases a
concrete layer behind callbacks for output shape, variables, and forward
execution; `Loss[T]` provides a common loss interface. These adapters let the
sequential model and optimizers work with multiple implementations.

This package is mainly useful when implementing custom layers or integrating
new components. Most application code can use constructors from
[`layers`](../layers/README.md), [`loss`](../loss/README.md), and
[`models`](../models/README.md).
