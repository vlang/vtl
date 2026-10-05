# `vtl.nn.gates`

Gates register reverse-mode backward rules with the autograd context. A gate
must return one gradient per parent, in the same order used during caching,
and preserve each parent's shape.

| Submodule | Rules |
| --- | --- |
| [`activation`](activation/) | ReLU, sigmoid, tanh, softmax, Leaky ReLU, ELU, GELU, Swish, Mish |
| [`layers`](layers/) | Linear, flatten, input, max-pooling, dropout, and LSTM gates |
| [`loss`](loss/) | MSE, extra losses, sigmoid cross entropy, and softmax cross entropy |

These are implementation adapters for [`layers`](../layers/README.md) and
[`loss`](../loss/README.md); application code should construct those public
components instead of creating gates directly. Gate coverage is narrower than
the complete tensor API. Check the corresponding gate tests before assuming a
gradient path is supported. CUDA and Vulkan variants are compile-time-specific
and experimental; see [device memory notes](../../docs/DEVICE_MEMORY.md).
