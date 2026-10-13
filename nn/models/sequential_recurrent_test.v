module models

import vtl
import vtl.autograd
import vtl.nn.layers

fn test_conv1d_sequential_forward_backward() ! {
	ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([1, 5])
	nn.conv1d(2, 3, layers.Conv1DConfig{ padding: 1 })
	mut input := ctx.variable(vtl.from_array([0.1, 0.2, 0.3, 0.4, 0.5], [1, 1, 5])!)
	mut output := nn.forward(input)!
	assert output.value.shape == [1, 2, 5]
	output.backprop()!
	assert input.grad.shape == input.value.shape
	assert nn.info.layers[1].variables().len == 2
}

fn test_gru_sequential_forward_backward() ! {
	ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([2, 1, 2])
	nn.gru(2, 3)
	mut input := ctx.variable(vtl.from_array([0.2, -0.1, 0.4, 0.3], [2, 1, 2])!)
	mut output := nn.forward(input)!
	assert output.value.shape == [2, 1, 3]
	assert nn.info.layers[1].variables().len == 4
	output.backprop()!
	assert input.grad.shape == input.value.shape
	for parameter in nn.info.layers[1].variables() {
		mut grad_norm := f64(0)
		for value in parameter.grad.to_array() {
			grad_norm += value * value
		}
		assert grad_norm > 0, 'Sequential GRU backward must update each parameter'
	}
}

fn test_sequential_can_stack_gru_layers() ! {
	ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([2, 1, 2])
	nn.gru(2, 3)
	nn.gru(3, 2)
	mut input := ctx.variable(vtl.from_array([0.2, -0.1, 0.4, 0.3], [2, 1, 2])!)
	mut output := nn.forward(input)!
	assert output.value.shape == [2, 1, 2]
	assert nn.info.layers[1].variables().len == 4
	assert nn.info.layers[2].variables().len == 4
	output.backprop()!
	assert input.grad.shape == input.value.shape
	for layer_index in 1 .. 3 {
		for parameter in nn.info.layers[layer_index].variables() {
			assert parameter.grad.shape == parameter.value.shape
			assert parameter.grad.to_array().any(it != 0), 'stacked GRU parameter gradient must be non-zero'
		}
	}
}
