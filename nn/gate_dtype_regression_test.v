module nn

import vtl
import vtl.autograd
import vtl.nn.models

fn run_sequential_mse_backprop_f64() ! {
	ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([2])
	model.linear(1)
	model.mse_loss()
	x := ctx.variable(vtl.ones[f64]([1, 2]))
	y := vtl.zeros[f64]([1, 1])
	pred := model.forward(x)!
	mut loss := model.loss(pred, y)!
	loss.backprop()!
}

fn run_sequential_mse_backprop_f32() ! {
	ctx := autograd.ctx[f32]()
	mut model := models.sequential_from_ctx[f32](ctx)
	model.input([2])
	model.linear(1)
	model.mse_loss()
	x := ctx.variable(vtl.ones[f32]([1, 2]))
	y := vtl.zeros[f32]([1, 1])
	pred := model.forward(x)!
	mut loss := model.loss(pred, y)!
	loss.backprop()!
}

fn run_sequential_relu_softmax_backprop_f64() ! {
	ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([2])
	model.linear(2)
	model.relu()
	model.softmax()
	model.mse_loss()
	x := ctx.variable(vtl.ones[f64]([1, 2]))
	y := vtl.zeros[f64]([1, 2])
	pred := model.forward(x)!
	mut loss := model.loss(pred, y)!
	loss.backprop()!
}

fn run_sequential_relu_softmax_backprop_f32() ! {
	ctx := autograd.ctx[f32]()
	mut model := models.sequential_from_ctx[f32](ctx)
	model.input([2])
	model.linear(2)
	model.relu()
	model.softmax()
	model.mse_loss()
	x := ctx.variable(vtl.ones[f32]([1, 2]))
	y := vtl.zeros[f32]([1, 2])
	pred := model.forward(x)!
	mut loss := model.loss(pred, y)!
	loss.backprop()!
}

fn test_nn_backprop_f64_then_f32() ! {
	run_sequential_mse_backprop_f64()!
	run_sequential_mse_backprop_f32()!
	run_sequential_relu_softmax_backprop_f64()!
	run_sequential_relu_softmax_backprop_f32()!
}

fn test_nn_backprop_f32_then_f64() ! {
	run_sequential_mse_backprop_f32()!
	run_sequential_mse_backprop_f64()!
	run_sequential_relu_softmax_backprop_f32()!
	run_sequential_relu_softmax_backprop_f64()!
}
