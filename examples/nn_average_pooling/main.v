module main

import vtl
import vtl.autograd
import vtl.nn.layers

fn main() {
	ctx := autograd.ctx[f64]()
	input_tensor := vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1, 1, 3, 3])!
	input := ctx.variable(input_tensor)
	pool := layers.avgpool2d_layer[f64](ctx, [1, 3, 3], [2, 2], [0, 0], [1, 1])
	mut output := pool.forward(input)!
	output.backprop()!

	println('Output shape: ${output.value.shape}')
	println('Output values: ${output.value.to_array()}')
	println('Input gradient: ${input.grad.to_array()}')
}
