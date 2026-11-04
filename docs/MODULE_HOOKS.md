# Module hooks

Module hooks observe, and optionally replace, the values that flow through a neural-network
layer during a forward pass and during backpropagation. They are how you inspect activations and
gradients, or collect diagnostics, without copying a model or changing a layer. Implemented in
`nn/hooks`, from issue [#808](https://github.com/vlang/vtl/issues/808).

## Usage

```v
import vtl.nn.hooks
import vtl.nn.layers

wrapped, recorder := hooks.with_hooks[f64](layers.relu_layer[f64](ctx, [4]))
handle := recorder.add_forward_hook(my_forward_hook)
output := wrapped.forward(input)!
handle.remove()
```

`with_hooks` returns:

- `wrapped`: a `types.Layer[T]` you use wherever the original layer was used, including inside
  a `Sequential` or nested in another wrapper.
- `recorder`: a `hooks.Hooks[T]` with `add_forward_pre_hook`, `add_forward_hook` and
  `add_backward_hook`. Each returns a `HookHandle[T]`, and `handle.remove()` unregisters it.

## Contract

| Area | Behaviour |
|------|-----------|
| Forward pre-hook | Runs before the layer, in registration order. Each receives the input produced by the previous hook and may return a replacement. |
| Layer | Runs with the input the last pre-hook returned. |
| Forward hook | Runs after the layer, in registration order. Each receives the input the layer saw and the output of the previous hook, and may return a replacement output. |
| Backward hook | Runs during backpropagation, in registration order. Each receives the gradient produced by the previous hook and may return a replacement gradient. The gradient is with respect to the module output. |
| Gradients disabled | Backward hooks attach only when `Variable.is_grad_needed()` holds, that is when `requires_grad` is set and the context is not `no_grad`. Forward hooks run either way. |
| Errors | A hook error in forward stops the pass and is returned unchanged. A hook error in backward stops backpropagation and is returned from `backprop`. |
| Removal | `remove()` takes the hook out of the registry, so later passes do not run it. A hook that is already running is not interrupted. Calling `remove()` again has no effect. |
| Order between kinds | Forward pre-hooks, the layer, forward hooks, then backward hooks in reverse graph order. |
| Parameters | Hooks are not variables and are not part of `variables()`. Serialization of the wrapped layer is unchanged. |

## Decisions

- **Wrapper, not a change to `types.Layer`.** The wrapper implements `types.Layer` by
  delegation. The Layer type, existing layers, and `Sequential` keep their current API, so
  nothing breaks. The cost is one extra indirection per call.
- **Replacement semantics.** Hooks may return a new tensor or variable. A hook that only
  observes returns its argument unchanged.
- **Gradient-enabled detection uses the framework's own rule.** `is_grad_needed` is the same
  check the autograd engine uses, so hooks and gates agree on when gradients are tracked.
- **Backward hooks run once per module output.** They are attached as one identity node
  between the module output and its consumers. Hooks inside one module run in registration
  order, and each sees the previous hook's gradient.

## Accelerator behaviour

The hooks run on the host and observe tensors through `to_array`, so they work with the CPU
path and with GPU-backed tensors. Hooks that call GPU kernels themselves are not covered by the
tests. CPU behaviour is the documented contract.

## Example

`examples/nn_module_hooks` records the activations and the gradient of a ReLU layer, and shows
that removing a hook stops it from printing.
