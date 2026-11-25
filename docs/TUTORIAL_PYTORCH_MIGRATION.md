# PyTorch to VTL migration

This guide maps common CPU training patterns to VTL. VTL uses generic V
numeric types, so choose `f32` or `f64` explicitly when creating the context
and tensors.

## Tensor and autograd

PyTorch:

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
loss = (x * x).sum()
loss.backward()
```

VTL:

```v
ctx := autograd.ctx[f64]()
x := ctx.variable(vtl.from_array([1.0, 2.0, 3.0], [3])!, requires_grad: true)
loss := x.multiply(x)!.sum()!
loss.backprop()!
```

See [the autograd tutorial](./TUTORIAL_AUTOGRAD.md) for graph lifetime and
supported differentiable operations.

## Modules, losses, and optimization

Use `nn.models.sequential` for a compact CPU model:

```v
mut model := models.sequential[f64]()
model.input([4])
model.linear(8)
model.relu()
model.linear(2)
criterion := loss.cross_entropy_loss[f64]()
```

VTL's layer and optimizer APIs are documented in the
[neural-network tutorial](./TUTORIAL_NEURAL_NETWORKS.md) and
[optimizer tutorial](./TUTORIAL_OPTIMIZERS.md). The current CPU path supports
reverse-mode autograd, common classification/regression losses, parameter
groups, and learning-rate schedulers. CUDA/Vulkan/OpenCL support is selective;
consult the [PyTorch parity tracker](./PYTORCH_PARITY.md) before assuming
device or backward parity.

## Batching and functional transforms

`vtl.autograd.vmap` is useful for per-sample functions and batched Jacobian
building blocks:

```v
batched := autograd.vmap[f64](fn (sample &autograd.Variable[f64]) !&autograd.Variable[f64] {
	return sample.add(sample)
}, x, 0, 0)!
```

For tensor-only transforms, use `vtl.vmap`; see the
[functional transforms tutorial](./TUTORIAL_FUNCTIONAL_TRANSFORMS.md).

## Checkpoints and inference

VTL's checkpoint and inference workflow is covered by
[TUTORIAL_INFERENCE_CHECKPOINT.md](./TUTORIAL_INFERENCE_CHECKPOINT.md). Keep
training and inference on CPU unless the parity tracker documents the selected
backend path for the model's layers and gradients.

## Intentional differences

VTL does not currently promise full `torch.nn`, `torch.func`, dtype/device,
distributed-training, sparse, or accelerator parity. Unsupported behavior
should fail explicitly; the parity tracker is the source of truth for current
coverage and the linked issue numbers identify active work.
