module layers

import vtl
import vtl.autograd

fn test_conv_transpose2d_layer_forward_and_backward() ! {
	ctx := autograd.ctx[f64]()
	layer := conv_transpose2d_layer[f64](ctx, 1, 1, [2, 2], ConvTranspose2DConfig{},
		[1, 1, 1])
	assert layer.output_shape() == [1, 2, 2]
	variables := layer.variables()
	mut input := ctx.variable(vtl.from_array([1.0], [1, 1, 1, 1])!)
	variables[0].value.set_nth(0, 1.0)
	variables[0].value.set_nth(1, 2.0)
	variables[0].value.set_nth(2, 3.0)
	variables[0].value.set_nth(3, 4.0)
	variables[1].value.set_nth(0, 0.5)
	output := layer.forward(input)!
	assert output.value.to_array() == [1.5, 2.5, 3.5, 4.5]
	output.backprop()!
	assert input.grad.to_array() == [10.0]
	assert variables[0].grad.to_array() == [1.0, 1.0, 1.0, 1.0]
	assert variables[1].grad.to_array() == [4.0]
}
