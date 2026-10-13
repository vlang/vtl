module models

import vtl
import vtl.autograd
import vtl.nn.layers
import math
import rand

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

fn test_rms_norm_in_sequential_forward_and_backward() ! {
	ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([2])
	nn.rms_norm([2], layers.RMSNormConfig{
		eps: 1e-5
	})
	input := ctx.variable(vtl.from_array([3.0, 4.0, 5.0, 12.0], [2, 2])!)
	output := nn.forward(input)!
	assert output.value.shape == [2, 2]
	assert nn.info.layer_types[1] == 'RMSNormLayer'
	mut loss := output.sum()!
	loss.backprop()!
	assert input.grad.shape == [2, 2]
	assert nn.info.layers[1].variables().len == 1
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

fn test_sequential_batchnorm2d_forward_and_backward() ! {
	ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([1, 2, 2])
	nn.batchnorm2d(1, layers.BatchNorm2DConfig{})
	assert nn.info.layer_types[1] == 'BatchNorm2DLayer'
	input := ctx.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])!)
	output := nn.forward(input)!
	assert output.value.shape == [1, 1, 2, 2]
	// Training mode (default): channel mean 2.5, biased var 1.25.
	inv_std := 1.0 / math.sqrt(1.25 + 1e-5)
	assert math.abs(output.value.get_nth(0) - (1.0 - 2.5) * inv_std) < 1e-12
	mut loss := output.sum()!
	loss.backprop()!
	assert input.grad.shape == [1, 1, 2, 2]
	assert nn.info.layers[1].variables().len == 2
	// Eval mode uses the running statistics updated during training.
	nn.eval()
	eval_output := nn.forward(input)!
	assert eval_output.value.to_array() != output.value.to_array()
}

fn test_sequential_batchnorm3d_forward_and_backward() ! {
	ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([1, 1, 2, 2])
	nn.batchnorm3d(1, layers.BatchNorm3DConfig{})
	assert nn.info.layer_types[1] == 'BatchNorm3DLayer'
	input := ctx.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 2, 2])!)
	output := nn.forward(input)!
	assert output.value.shape == [1, 1, 1, 2, 2]
	inv_std := 1.0 / math.sqrt(1.25 + 1e-5)
	assert math.abs(output.value.get_nth(0) - (1.0 - 2.5) * inv_std) < 1e-12
	mut loss := output.sum()!
	loss.backprop()!
	assert input.grad.shape == [1, 1, 1, 2, 2]
	assert nn.info.layers[1].variables().len == 2
	nn.eval()
	eval_output := nn.forward(input)!
	assert eval_output.value.to_array() != output.value.to_array()
}
