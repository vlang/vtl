module models

import vtl
import vtl.autograd
import vtl.nn.optimizers
import vtl.nn.types

struct RegisteredStateLayerModeSpy {
mut:
	modes []bool
}

fn registered_state_layer_output_shape(ptr voidptr) []int {
	return [1]
}

fn registered_state_layer_variables(ptr voidptr) []voidptr {
	return []voidptr{}
}

fn registered_state_layer_forward(ptr voidptr, input voidptr) !voidptr {
	return input
}

fn registered_state_layer_forward_mode(ptr voidptr, input voidptr, training bool) !voidptr {
	mut state := unsafe { &RegisteredStateLayerModeSpy(ptr) }
	state.modes << training
	return input
}

fn test_nnc() {
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.sigmoid()
	assert nn.info.layers.len == 2
	assert nn.info.layers[0].output_shape() == [1, 2]
	assert nn.info.layers[1].output_shape() == [1, 2]
}

fn test_sequential_integrates_registered_custom_module_state() ! {
	ctx := autograd.ctx[f64]()
	parameter := ctx.variable(vtl.from_array([2.0], [1])!)
	buffer := vtl.from_array([3.0], [1])!
	mut root := types.module[f64]()
	mut child := types.module[f64]()
	root.register_parameter('weight', parameter)!
	child.register_buffer('running_mean', buffer)!
	root.register_module('normalizer', child)!

	mut mode_spy := &RegisteredStateLayerModeSpy{}
	custom_layer := types.layer_with_mode[f64](voidptr(mode_spy),
		registered_state_layer_output_shape, registered_state_layer_variables,
		registered_state_layer_forward, registered_state_layer_forward_mode)
	layer := types.layer_with_module_state(custom_layer, root)
	mut model := sequential_with_layers[f64]([layer])

	mut optimizer := optimizers.adam_optimizer[f64](learning_rate: 0.01)
	optimizer.build_params(model.info.layers)
	assert optimizer.params.len == 1
	assert optimizer.params[0] == parameter
	named_buffers := root.named_buffers()
	assert named_buffers.len == 1
	assert named_buffers[0].value == buffer

	input := ctx.variable(vtl.from_array([1.0], [1])!)
	model.eval()
	assert !root.training
	assert !child.training
	model.forward(input)!
	model.train()
	assert root.training
	assert child.training
	model.forward(input)!
	assert mode_spy.modes == [false, true]
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

fn test_sequential_no_grad_does_not_record_inference_graph() ! {
	mut ctx := autograd.ctx[f64]()
	mut nn := sequential_from_ctx[f64](ctx)
	nn.input([2])
	nn.linear(1)
	nn.eval()

	input := ctx.variable(vtl.from_array([1.0, 2.0], [1, 2])!)
	ctx.no_grad = true
	output := nn.forward(input)!
	assert ctx.len() == 0
	assert !output.is_grad_needed()

	ctx.no_grad = false
	assert !output.is_grad_needed()

	mut prediction := output.sum()!
	assert ctx.len() == 0
	assert !prediction.is_grad_needed()
	prediction.backprop()!
	assert input.grad.to_array() == [0.0, 0.0]
	for layer in nn.info.layers {
		for parameter in layer.variables() {
			assert parameter.grad.to_array().all(it == 0.0)
		}
	}
}
