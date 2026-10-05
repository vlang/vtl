# `vtl.nn.internal`

Internal numerical kernels and helpers shared by VTL's neural-network modules:
initialization, activation math, losses, optimizer updates, convolution and
pooling kernels, recurrent operations, and normalization. These functions are
implementation details and may change without compatibility guarantees.

Use [`layers`](../layers/README.md), [`loss`](../loss/README.md), and
[`optimizers`](../optimizers/README.md) from application code. Backend-specific
files are selected by V compile-time conditions; preserve CPU behavior when
changing accelerator code.
