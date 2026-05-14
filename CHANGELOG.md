# Changelog

## Unreleased

### Neural Networks

- **GRU layer** — Gated Recurrent Unit implementation (PyTorch/CuDNN compatible).
  3 gates (reset, update, new), fewer parameters than LSTM, often trains faster.
  Usage: `layers.gru_layer[f64](ctx, input_size, hidden_size)`
- **LSTM fix** — Complete rewrite with proper cell state and 4-gate (i, f, g, o) architecture.
  Previously missing cell state propagation.
- **Runtime backend dispatch** — NN layers (Linear, ReLU, Sigmoid, Tanh, Conv2D, MaxPool2D,
  BatchNorm, LayerNorm, Embedding, Attention, Softmax) now dispatch to GPU backends at runtime
  via `VTL_BACKEND` environment variable.

### Autograd

- Added `sin()`, `cos()`, `tan()` differentiable operations on Variables.
- Correct gradient flow through all unary gates.

### GPU Backends

- **Unified compute API** — VCL paths migrated to `vsl.compute` interface.
- **Vulkan multi-head attention** — Forward pass runs on Vulkan when available.
- **Runtime backend selection** — `VTL_BACKEND=vulkan|vcl|cpu|auto` controls dispatch
  without recompilation. `VTL_BACKEND_STRICT=1` fails if preferred backend is unavailable.

### Examples

- `nn_regression_sine_plot/` — MLP learns sin(x), plots predicted vs true + loss curve.
- `nn_training_metrics_plot/` — XOR classifier with loss/accuracy visualization.
- `autograd_gradient_plot/` — Autograd computes sin'(x) = cos(x), plots comparison.
- `vtl_plot_scatter_colorscale/` — Scatter plot with colorscale using vsl.plot.

### Documentation

- **TUTORIAL.md** rewritten as a narrative getting-started guide (zero to trained model).
- **TUTORIAL_NEURAL_NETWORKS.md** — added "Visualizing Training with vsl.plot" section
  and "Recurrent layers: LSTM and GRU" comparison.
- **TUTORIAL_GPU_BACKENDS.md** — documents runtime backend selection and layer coverage.
- **TUTORIAL_OPTIMIZERS.md** — covers Adam, AdamW, RMSProp, AdaGrad, SGD + LR schedulers.
- **TUTORIAL_AUTOGRAD.md** — unary gates, reduction gates, shape gates documented.

### CI / Build

- All non-Vulkan CI lanes green (Linux + macOS, with and without `--prod`).
- Fixed macOS OpenCL linking (`-d vsl_vcl_dlopencl` applied globally).
- Fixed GCC LTO overflow in `--prod` builds (`-cflags -fno-lto`).
- Vulkan tests moved to external module to avoid V's expr depth limit (40 levels).
- 60-minute timeout on Vulkan CI jobs to prevent runner reclamation.

### Breaking Changes

- LSTM layer now requires cell state (signature changed to return `(output, h_n, c_n)`).
- Vulkan layer tests moved from `nn/layers/` to `tests/vulkan_layers/` module.
