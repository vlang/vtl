module autograd_tests

import vtl
import vtl.autograd

fn test_clamp_forward_and_backward_mask() ! {
	context := autograd.ctx[f64]()
	input := context.variable(vtl.from_1d([-2.0, -1.0, 0.0, 1.0, 2.0])!)
	mut output := input.clamp(-1.0, 1.0)!
	assert output.value.array_equal(vtl.from_1d([-1.0, -1.0, 0.0, 1.0, 1.0])!)
	output.backprop()!
	assert input.grad.array_equal(vtl.from_1d([0.0, 1.0, 1.0, 1.0, 0.0])!)
}
