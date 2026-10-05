module layers

import vtl
import vtl.autograd

fn test_conv1d_layer_forward_and_backward() ! {
	ctx := autograd.ctx[f64]()
	layer := conv1d_layer[f64](ctx, 1, 2, 3, Conv1DConfig{ padding: 1 }, 5)
	assert layer.output_shape() == [2, 5]
	assert layer.variables().len == 2
	mut input := ctx.variable(vtl.from_array([0.1, 0.2, 0.3, 0.4, 0.5], [1, 1, 5])!)
	mut output := layer.forward(input)!
	assert output.value.shape == [1, 2, 5]
	output.backprop()!
	assert input.grad.shape == input.value.shape
	assert layer.variables()[0].grad.shape == layer.variables()[0].value.shape
	assert layer.variables()[1].grad.shape == layer.variables()[1].value.shape
}

fn test_conv1d_layer_rejects_wrong_channel_count() {
	ctx := autograd.ctx[f64]()
	layer := conv1d_layer[f64](ctx, 2, 3, 2, Conv1DConfig{}, 5)
	input := ctx.variable(vtl.zeros[f64]([1, 1, 5]))
	layer.forward(input) or {
		assert err.msg().contains('shape')
		return
	}
	assert false, 'expected mismatched channels to fail'
}
