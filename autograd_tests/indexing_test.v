module autograd_tests

import vtl
import vtl.autograd

fn test_take_along_axis_backward_accumulates_repeated_indices() ! {
	mut ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [2, 3])!)
	indices := vtl.from_array[int]([1, 1, 0, 2], [2, 2])!

	mut selected := input.take_along_axis(indices, 1)!
	selected.backprop()!
	assert input.grad.to_array() == [0.0, 2.0, 0.0, 1.0, 0.0, 1.0]
}

fn test_take_along_axis_backward_reduces_broadcast_dimensions() ! {
	mut ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([10.0, 20.0, 30.0], [1, 3])!)
	indices := vtl.from_array[int]([1, 1, 2, 1], [2, 2])!

	mut selected := input.take_along_axis(indices, 1)!
	assert selected.value.shape == [2, 2]
	selected.backprop()!
	assert input.grad.to_array() == [0.0, 3.0, 1.0]
}

fn test_take_along_axis_backward_supports_negative_axis_and_indices() ! {
	mut ctx := autograd.ctx[f32]()
	input := ctx.variable(vtl.from_array([1.0, 2.0, 3.0], [1, 3])!)
	indices := vtl.from_array[int]([-1, 0], [1, 2])!

	mut selected := input.take_along_axis(indices, -1)!
	selected.backprop()!
	assert input.grad.to_array() == [f32(1.0), 0.0, 1.0]
}

fn test_take_along_axis_backward_uses_forward_index_snapshot() ! {
	mut ctx := autograd.ctx[f64]()
	input := ctx.variable(vtl.from_array([10.0, 20.0, 30.0], [1, 3])!)
	mut indices := vtl.from_array[int]([1, 1], [1, 2])!
	mut selected := input.take_along_axis(indices, 1)!
	indices.fill(0)

	selected.backprop()!
	assert input.grad.to_array() == [0.0, 2.0, 0.0]
}
