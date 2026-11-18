# CPU Sequential inference checkpoints

VTL can save and load a small, versioned JSON checkpoint for CPU inference:

```v ignore
model.save_inference_checkpoint('model.json')!
mut restored := models.sequential_from_ctx[f64](autograd.ctx[f64]())
restored.input([2])
restored.linear(1)
restored.load_inference_checkpoint('model.json')!
restored.eval()
```

The initial format supports `f32`/`f64` `Sequential` models built from
`input`, `linear`, and `relu` layers. Build the destination model first; loading
checks the checkpoint version, dtype, ordered layer types/configurations,
output and parameter shapes, and finite parameter values before changing any
weights. The format is JSON data only and does not instantiate layers or execute
code from the file.

This API is for inference weights, not training resume. It does not save
optimizer state, loss configuration, autograd graphs, or training metadata.
Existing `save_checkpoint` and optimizer-specific training checkpoint APIs
remain separate. Other layer types, accelerator-backed models, dtype conversion,
and checkpoint migration are not supported.

Run the end-to-end train/save/load/infer example from `~/.vmodules`:

```sh
v run ./vtl/examples/sequential_inference_checkpoint.v
```
