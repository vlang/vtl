module nn

import vtl
import vtl.autograd
import vtl.nn.models

fn test_sequential_f32_mse_forward_backprop() ! {
	ctx := autograd.ctx[f32]()
	mut model := models.sequential_from_ctx[f32](ctx)
	model.input([4])
	model.linear(3)
	model.mse_loss()
	x := vtl.ones[f32]([1, 4])
	y := vtl.zeros[f32]([1, 3])
	v := ctx.variable(x)
	pred := model.forward(v)!
	mut loss := model.loss(pred, y)!
	loss.backprop()!
}
