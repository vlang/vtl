module models

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.types
import math
import rand

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

fn test_group_norm_in_sequential_forward_and_backward() ! {
	ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([2, 2])
	nn.group_norm(1, layers.GroupNormConfig{})
	input := ctx.variable(vtl.from_array([1.0, 2.0, 4.0, 0.0], [1, 2, 2])!)
	output := nn.forward(input)!
	assert output.value.shape == [1, 2, 2]
	assert nn.info.layer_types[1] == 'GroupNormLayer'
	mut loss := output.sum()!
	loss.backprop()!
	assert input.grad.shape == [1, 2, 2]
	assert nn.info.layers[1].variables().len == 2
}

fn test_sequential_dropout_is_identity_in_eval_mode() ! {
	rand.seed([u32(42), u32(0)])
	ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	assert nn.training
	nn.input([4])
	nn.dropout(0.5)
	nn.eval()
	input := ctx.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0], [1, 4])!, requires_grad: false)
	output := nn.forward(input)!
	assert output.value.to_array() == [1.0, 2.0, 3.0, 4.0]

	nn.train()
	training_output := nn.forward(input)!
	assert training_output.value.to_array() != [1.0, 2.0, 3.0, 4.0]
	nn.eval()
	assert !nn.training
}

fn test_sequential_dropout_probability_one_zeros_output_and_gradient() ! {
	ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([3])
	nn.dropout(1.0)
	mut input := ctx.variable(vtl.from_array([2.0, -3.0, 4.0], [1, 3])!)
	mut output := nn.forward(input)!
	assert output.value.to_array() == [0.0, 0.0, 0.0]
	output.backprop()!
	assert input.grad.to_array() == [0.0, 0.0, 0.0]
}

fn test_sequential_batchnorm_mode_is_independent_of_input_gradients() ! {
	ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([2])
	nn.batchnorm1d(2, layers.BatchNorm1DConfig{})
	input := ctx.variable(vtl.from_array([1.0, 3.0, 5.0, 7.0], [2, 2])!, requires_grad: false)

	training_output := nn.forward(input)!
	assert training_output.requires_grad
	assert math.abs(training_output.value.get_nth(0) + 0.99999875) < 1e-6
	assert math.abs(training_output.value.get_nth(2) - 0.99999875) < 1e-6
	mut training_loss := training_output.sum()!
	training_loss.backprop()!
	batchnorm_vars := nn.info.layers[1].variables()
	assert batchnorm_vars[1].grad.to_array() == [2.0, 2.0]
	// The model's batch normalization layer updates running stats in training
	// mode even when the input itself does not need gradients.
	nn.eval()
	eval_output := nn.forward(input)!
	assert eval_output.requires_grad
	assert eval_output.value.to_array() != training_output.value.to_array()
	assert math.abs(eval_output.value.get_nth(0) - 0.6139) < 1e-3
	mut eval_loss := eval_output.sum()!
	eval_loss.backprop()!
	assert batchnorm_vars[1].grad.to_array() == [4.0, 4.0]
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
