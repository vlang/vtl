module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types

// GroupNormConfig controls affine parameters and numerical stability.
@[params]
pub struct GroupNormConfig {
pub:
	eps    f64  = 1e-5
	affine bool = true
}

// GroupNormLayer normalizes channel groups independently for each sample.
pub struct GroupNormLayer[T] {
pub:
	input_shape []int
	num_groups  int
	eps         f64
pub mut:
	gamma &autograd.Variable[T] = unsafe { nil }
	beta  &autograd.Variable[T] = unsafe { nil }
}

// group_norm_layer creates a GroupNorm layer. input_shape excludes batch and
// uses channel-first order: [channels, ...spatial dimensions].
pub fn group_norm_layer[T](ctx &autograd.Context[T], input_shape []int, num_groups int, config GroupNormConfig) types.Layer[T] {
	channels := if input_shape.len > 0 { input_shape[0] } else { 0 }
	mut gamma := unsafe { nil }
	mut beta := unsafe { nil }
	if config.affine {
		gamma = ctx.variable(vtl.ones[T]([channels]))
		beta = ctx.variable(vtl.zeros[T]([channels]))
	}
	layer := &GroupNormLayer[T]{
		input_shape: input_shape.clone()
		num_groups:  num_groups
		eps:         config.eps
		gamma:       gamma
		beta:        beta
	}
	return types.layer[T](voidptr(layer), group_norm_layer_output_shape_dispatch[T],
		group_norm_layer_variables_dispatch[T], group_norm_layer_forward_dispatch[T])
}

// output_shape returns the feature shape, excluding batch.
pub fn (layer &GroupNormLayer[T]) output_shape() []int {
	return layer.input_shape.clone()
}

// variables returns affine trainable parameters, if enabled.
pub fn (layer &GroupNormLayer[T]) variables() []&autograd.Variable[T] {
	mut variables := []&autograd.Variable[T]{}
	if layer.gamma != unsafe { nil } { variables << layer.gamma }
	if layer.beta != unsafe { nil } { variables << layer.beta }
	return variables
}

// forward applies GroupNorm and records its backward operation when needed.
pub fn (layer &GroupNormLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	if input.value.shape.len != layer.input_shape.len + 1 || input.value.shape[1..] != layer.input_shape {
		return error('GroupNorm: input feature shape does not match layer input_shape')
	}
	gamma_value := if layer.gamma != unsafe { nil } {
		layer.gamma.value
	} else {
		unsafe { &vtl.Tensor[T](nil) }
	}
	beta_value := if layer.beta != unsafe { nil } {
		layer.beta.value
	} else {
		unsafe { &vtl.Tensor[T](nil) }
	}
	output := internal.group_norm_forward[T](input.value, layer.num_groups, gamma_value, beta_value,
		layer.eps)!
	mut result := input.context.variable(output,
		requires_grad: input.requires_grad || (layer.gamma != unsafe { nil } && layer.gamma.requires_grad)
			|| (layer.beta != unsafe { nil } && layer.beta.requires_grad)
	)
	if result.requires_grad {
		gate := group_norm_gate[T](input.value, layer.num_groups, gamma_value, beta_value, layer.eps)
		if layer.gamma != unsafe { nil } && layer.beta != unsafe { nil } {
			gate.cache(mut result, input, layer.gamma, layer.beta)!
		} else if layer.gamma != unsafe { nil } {
			gate.cache(mut result, input, layer.gamma)!
		} else if layer.beta != unsafe { nil } {
			gate.cache(mut result, input, layer.beta)!
		} else {
			gate.cache(mut result, input)!
		}
	}
	return result
}

fn group_norm_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&GroupNormLayer[T](layer)).output_shape() }
}

fn group_norm_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&GroupNormLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn group_norm_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&GroupNormLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

// GroupNormGate stores the tensors needed to compute GroupNorm gradients.
pub struct GroupNormGate[T] {
	input      &vtl.Tensor[T] = unsafe { nil }
	num_groups int
	gamma      &vtl.Tensor[T] = unsafe { nil }
	beta       &vtl.Tensor[T] = unsafe { nil }
	eps        f64
}

// group_norm_gate creates a backward gate for GroupNorm.
pub fn group_norm_gate[T](input &vtl.Tensor[T], num_groups int, gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) &GroupNormGate[T] {
	return &GroupNormGate[T]{
		input:      input
		num_groups: num_groups
		gamma:      gamma
		beta:       beta
		eps:        eps
	}
}

// backward returns input and present affine parameter gradients.
pub fn (gate &GroupNormGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return internal.group_norm_backward[T](payload.variable.grad, gate.input, gate.num_groups,
		gate.gamma, gate.beta, gate.eps)
}

fn group_norm_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&GroupNormGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers this gate with its input and trainable parameters.
pub fn (gate &GroupNormGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	expected_len := 1 + if gate.gamma != unsafe { nil } { 1 } else { 0 } + if gate.beta != unsafe { nil } {
		1
	} else {
		0
	}
	if args.len != expected_len {
		return error('GroupNormGate.cache: unexpected number of variable arguments')
	}
	mut parents := []&autograd.Variable[T]{cap: expected_len}
	for index, arg in args {
		match arg {
			autograd.Variable[T] { parents << arg }
			else { return error('GroupNormGate.cache: argument ${index} must be a Variable') }
		}
	}
	result.grad = vtl.zeros_like[T](result.value)
	result.requires_grad = true
	autograd.register[T]('GroupNorm', voidptr(gate), group_norm_gate_backward_dispatch[T], result,
		parents)!
}
