module main

import os
import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers
import vtl.nn.types

struct AffineModuleState {
	weight &autograd.Variable[f64]
	bias   &autograd.Variable[f64]
mut:
	forward_modes []bool
}

fn affine_output_shape(_ voidptr) []int {
	return [1]
}

fn affine_variables(_ voidptr) []voidptr {
	return []voidptr{}
}

fn affine_forward(ptr voidptr, input voidptr) !voidptr {
	state := unsafe { &AffineModuleState(ptr) }
	input_variable := unsafe { &autograd.Variable[f64](input) }
	scaled := input_variable.multiply(state.weight)!
	return voidptr(scaled.add(state.bias)!)
}

fn affine_forward_mode(ptr voidptr, input voidptr, training bool) !voidptr {
	mut state := unsafe { &AffineModuleState(ptr) }
	state.forward_modes << training
	return affine_forward(ptr, input)
}

fn main() {
	ctx := autograd.ctx[f64]()
	weight := ctx.variable(vtl.from_1d([0.5])!)
	bias := ctx.variable(vtl.from_1d([0.0])!)
	running_mean := vtl.from_1d([0.0])!

	mut module := types.module[f64]()
	mut normalizer := types.module[f64]()
	module.register_parameter('weight', weight) or { panic(err) }
	module.register_parameter('bias', bias) or { panic(err) }
	normalizer.register_buffer('running_mean', running_mean) or { panic(err) }
	module.register_module('normalizer', normalizer) or { panic(err) }
	state := &AffineModuleState{
		weight: weight
		bias:   bias
	}
	callback_layer := types.layer_with_mode[f64](voidptr(state), affine_output_shape,
		affine_variables, affine_forward, affine_forward_mode)
	layer := types.layer_with_module_state(callback_layer, module)
	mut model := models.sequential_with_layers[f64]([layer])

	model.eval()
	assert !module.training
	assert !normalizer.training
	model.train()
	assert module.training
	assert normalizer.training

	mut optimizer := optimizers.adam_optimizer[f64](learning_rate: 0.01)
	optimizer.build_params(model.info.layers)
	assert optimizer.params.len == 2
	assert module.named_parameters().len == 2
	assert module.named_buffers().len == 1

	input := ctx.variable(vtl.from_1d([2.0])!)
	target := ctx.variable(vtl.from_1d([4.0])!, requires_grad: false)
	prediction := model.forward(input)!
	mut difference := prediction.subtract(target)!
	mut loss := difference.multiply(difference)!.sum()!
	loss.backprop()!
	assert state.forward_modes == [true]
	optimizer.update()!
	trained_weight := weight.value.to_array()
	trained_bias := bias.value.to_array()
	assert trained_weight != [0.5]
	assert trained_bias != [0.0]

	checkpoint_path := os.join_path(os.temp_dir(), 'vtl_custom_module_checkpoint.json')
	defer {
		os.rm(checkpoint_path) or {}
	}
	model.save_checkpoint(checkpoint_path, 1, f64(loss.value.get_nth(0)))!
	weight.value.set_nth(0, 10.0)
	bias.value.set_nth(0, 11.0)
	running_mean.set_nth(0, 12.0)
	model.load_weights(checkpoint_path)!
	assert weight.value.to_array() == trained_weight
	assert bias.value.to_array() == trained_bias
	assert running_mean.to_array() == [0.0]

	println('custom module trained; parameters, buffer, and mode state round-tripped')
}
