module layers

import vtl
import vtl.autograd

fn test_conv_transpose1d_layer_forward_and_backward() ! {
	ctx := autograd.ctx[f64]()
	layer := conv_transpose1d_layer[f64](ctx, 1, 1, 3, ConvTranspose1DConfig{}, 2)
	assert layer.output_shape() == [1, 4]
	assert layer.variables().len == 2
	variables := layer.variables()
	mut input := ctx.variable(vtl.from_array([1.0, 2.0], [1, 1, 2])!)
	variables[0].value.set_nth(0, 1.0)
	variables[0].value.set_nth(1, 2.0)
	variables[0].value.set_nth(2, 3.0)
	variables[1].value.set_nth(0, 0.5)
	output := layer.forward(input)!
	assert output.value.to_array() == [1.5, 4.5, 7.5, 6.5]
	output.backprop()!
	assert input.grad.to_array() == [6.0, 6.0]
	assert variables[0].grad.to_array() == [3.0, 3.0, 3.0]
	assert variables[1].grad.to_array() == [4.0]
}

fn test_conv_transpose1d_layer_rejects_wrong_channel_count() {
	ctx := autograd.ctx[f64]()
	layer := conv_transpose1d_layer[f64](ctx, 2, 3, 2, ConvTranspose1DConfig{}, 5)
	input := ctx.variable(vtl.zeros[f64]([1, 1, 5]))
	layer.forward(input) or {
		assert err.msg().contains('shape')
		return
	}
	assert false, 'expected mismatched channels to fail'
}
