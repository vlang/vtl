module nn

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models

// Conv2D gate must register input, weight, and bias as backprop parents.
fn test_conv2d_sequential_backprop_smoke() ! {
	mut ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([3, 8, 8])
	model.conv2d(3, 4, [3, 3], layers.Conv2DConfig{ padding: [1, 1] })
	model.flatten()
	model.linear(6)
	model.mse_loss()

	x := ctx.variable(vtl.ones[f64]([1, 3, 8, 8]), requires_grad: true)
	pred := model.forward(x)!
	mut loss := model.loss(pred, vtl.zeros[f64](pred.value.shape))!
	loss.backprop()!
	assert x.grad != unsafe { nil }
}
