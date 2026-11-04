# Module hooks example

Records the activations and gradients of a ReLU layer with forward and backward hooks, without
changing the layer. The hooks are registered on a wrapper returned by `hooks.with_hooks`, and
the wrapper is used wherever the layer would be used.

## Run

From `~/.vmodules` (never from inside the vtl checkout):

```sh
v run vtl/examples/nn_module_hooks/main.v
```

The contract (ordering, replacement, removal, no-gradient behaviour and errors) is documented
in `nn/hooks/hooks.v` and summarised in `docs/MODULE_HOOKS.md`.
