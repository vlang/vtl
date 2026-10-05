module models

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.types

fn test_nnc() {
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.sigmoid()
	assert nn.info.layers.len == 2
	assert nn.info.layers[0].output_shape() == [1, 2]
	assert nn.info.layers[1].output_shape() == [1, 2]
}

fn test_nn() {
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
}

fn test_new_sequential_losses() ! {
	c := autograd.ctx[f64]()
	pred := c.variable(vtl.from_array([1.0, -0.5], [2])!)
	target := vtl.from_array([0.0, -1.0], [2])!
	hinge_target := vtl.from_array([1.0, -1.0], [2])!
	focal_target := vtl.from_array([0.0, 1.0], [2])!
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})

	nn.l1_loss()
	mut l1 := nn.loss(pred, target)!
	assert l1.value.shape == [1]
	l1.backprop()!

	nn.hinge_loss()
	mut hinge := nn.loss(pred, hinge_target)!
	assert hinge.value.shape == [1]
	hinge.backprop()!

	nn.focal_loss()
	mut focal := nn.loss(pred, focal_target)!
	assert focal.value.shape == [1]
	focal.backprop()!
}

fn test_added_activation_layers_in_sequential() {
	mut nn := sequential[f64]()
	nn.input([3])
	nn.softplus()
	nn.selu()
	nn.hardswish()
	assert nn.info.layers.len == 4
	assert nn.info.layers[1].output_shape() == [3]
	assert nn.info.layers[2].output_shape() == [3]
	assert nn.info.layers[3].output_shape() == [3]
}

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
