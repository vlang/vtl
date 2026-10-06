module autograd_tests

import vtl
import vtl.autograd

fn test_matmul_f64_transposed_view_and_backward() {
	ctx := autograd.ctx[f64]()
	a := ctx.variable(vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!)
	b := ctx.variable(vtl.from_2d([[5.0, 6.0], [7.0, 8.0]])!.t()!)

	mut result := a.matmul(b)!
	assert result.value.array_equal(vtl.from_2d([[17.0, 23.0], [39.0, 53.0]])!)
	result.backprop()!

	assert a.grad.array_equal(vtl.from_2d([[12.0, 14.0], [12.0, 14.0]])!)
	assert b.grad.array_equal(vtl.from_2d([[4.0, 4.0], [6.0, 6.0]])!)
}

fn test_matmul_f32_keeps_dtype() {
	ctx := autograd.ctx[f32]()
	a := ctx.variable(vtl.from_2d[f32]([[f32(1.0), 2.0], [3.0, 4.0]])!)
	b := ctx.variable(vtl.from_2d[f32]([[f32(5.0), 6.0], [7.0, 8.0]])!)

	result := a.matmul(b)!
	assert result.value.array_equal(vtl.from_2d[f32]([[f32(19.0), 22.0], [43.0, 50.0]])!)
}
