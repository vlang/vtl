module autograd_tests

import vtl
import vtl.autograd

fn run_add_backprop_f64() ! {
	ctx := autograd.ctx[f64]()
	x := ctx.variable(vtl.from_1d([1.0, 2.0])!)
	y := ctx.variable(vtl.from_1d([3.0, 4.0])!)
	mut z := x.add(y)!
	z.backprop()!
	assert x.grad.array_equal(vtl.from_1d([1.0, 1.0])!)
}

fn run_add_backprop_f32() ! {
	ctx := autograd.ctx[f32]()
	x := ctx.variable(vtl.from_1d[f32]([f32(1.0), 2.0])!)
	y := ctx.variable(vtl.from_1d[f32]([f32(3.0), 4.0])!)
	mut z := x.add(y)!
	z.backprop()!
	assert x.grad.array_equal(vtl.from_1d[f32]([f32(1.0), 1.0])!)
}

fn run_unary_reshape_backprop_f64() ! {
	ctx := autograd.ctx[f64]()
	x := ctx.variable(vtl.from_1d([1.0, 2.0, 3.0, 4.0])!)
	y := x.sin()!
	mut z := y.reshape([2, 2])!
	z.backprop()!
	assert x.grad.shape == [4]
}

fn run_unary_reshape_backprop_f32() ! {
	ctx := autograd.ctx[f32]()
	x := ctx.variable(vtl.from_1d[f32]([f32(1.0), 2.0, 3.0, 4.0])!)
	y := x.sin()!
	mut z := y.reshape([2, 2])!
	z.backprop()!
	assert x.grad.shape == [4]
}

fn test_autograd_backprop_f64_then_f32() ! {
	run_add_backprop_f64()!
	run_add_backprop_f32()!
	run_unary_reshape_backprop_f64()!
	run_unary_reshape_backprop_f32()!
}

fn test_autograd_backprop_f32_then_f64() ! {
	run_add_backprop_f32()!
	run_add_backprop_f64()!
	run_unary_reshape_backprop_f32()!
	run_unary_reshape_backprop_f64()!
}
