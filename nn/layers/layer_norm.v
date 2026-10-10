module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types

// LayerNorm normalizes over the last D dimensions of the input.
// E.g. for input [..., D] it computes mean and variance over the last D dims.

// LayerNormConfig defines a public data structure for this module.

// LayerNormConfig defines a public data structure for this module.
@[params]
pub struct LayerNormConfig {
pub:
	eps    f64  = 1e-5
	affine bool = true
}

// LayerNormLayer defines a public data structure for this module.
pub struct LayerNormLayer[T] {
pub:
	normalized_shape []int
	eps              f64
pub mut:
	gamma &autograd.Variable[T] = unsafe { nil }
	beta  &autograd.Variable[T] = unsafe { nil }
}

// layer_norm_layer creates a LayerNormLayer.
// layer_norm_layer creates a LayerNormLayer.
pub fn layer_norm_layer[T](ctx &autograd.Context[T], normalized_shape []int, config LayerNormConfig) types.Layer[T] {
	mut gamma := unsafe { nil }
	mut beta := unsafe { nil }
	if config.affine {
		gamma = ctx.variable(vtl.ones[T](normalized_shape))
		beta = ctx.variable(vtl.zeros[T](normalized_shape))
	}
	layer := &LayerNormLayer[T]{
		normalized_shape: normalized_shape
		eps:              config.eps
		gamma:            gamma
		beta:             beta
	}
	return types.layer[T](voidptr(layer), layer_norm_layer_output_shape_dispatch[T],
		layer_norm_layer_variables_dispatch[T], layer_norm_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &LayerNormLayer[T]) output_shape() []int {
	return layer.normalized_shape
}

// variables exposes this operation as part of the public API.
pub fn (layer &LayerNormLayer[T]) variables() []&autograd.Variable[T] {
	mut variables := []&autograd.Variable[T]{}
	if layer.gamma != unsafe { nil } {
		variables << layer.gamma
	}
	if layer.beta != unsafe { nil } {
		variables << layer.beta
	}
	return variables
}

// forward exposes this operation as part of the public API.
pub fn (layer &LayerNormLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
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
	output := internal.layer_norm_forward_shape[T](input.value, gamma_value, beta_value,
		layer.normalized_shape, layer.eps)!
	mut result := input.context.variable(output,
		requires_grad: input.requires_grad || (layer.gamma != unsafe { nil } && layer.gamma.requires_grad)
			|| (layer.beta != unsafe { nil } && layer.beta.requires_grad)
	)
	if result.requires_grad {
		gate := layernorm_gate_with_shape[T](input.value, gamma_value, beta_value, layer.eps,
			layer.normalized_shape)
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

fn layer_norm_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&LayerNormLayer[T](layer)).output_shape() }
}

fn layer_norm_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&LayerNormLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn layer_norm_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&LayerNormLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

// LayerNormGate defines a public data structure for this module.
pub struct LayerNormGate[T] {
	input            &vtl.Tensor[T] = unsafe { nil }
	gamma            &vtl.Tensor[T] = unsafe { nil }
	beta             &vtl.Tensor[T] = unsafe { nil }
	eps              f64
	normalized_shape []int
}

// layernorm_gate exposes this operation as part of the public API.
pub fn layernorm_gate[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) &LayerNormGate[T] {
	normalized_shape := if gamma != unsafe { nil } {
		gamma.shape.clone()
	} else if beta != unsafe { nil } {
		beta.shape.clone()
	} else {
		input.shape.clone()
	}
	return layernorm_gate_with_shape[T](input, gamma, beta, eps, normalized_shape)
}

// layernorm_gate_with_shape creates a backward gate for trailing dimensions.
pub fn layernorm_gate_with_shape[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64, normalized_shape []int) &LayerNormGate[T] {
	return &LayerNormGate[T]{
		input:            input
		gamma:            gamma
		beta:             beta
		eps:              eps
		normalized_shape: normalized_shape.clone()
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &LayerNormGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return internal.layer_norm_backward_shape[T](payload.variable.grad, g.input, g.gamma, g.beta,
		g.normalized_shape, g.eps)
}

fn layer_norm_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&LayerNormGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &LayerNormGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	expected_len := 1 + if g.gamma != unsafe { nil } { 1 } else { 0 } + if g.beta != unsafe { nil } {
		1
	} else {
		0
	}
	if args.len != expected_len {
		return error('LayerNormGate.cache: expected ${expected_len} variable arguments')
	}
	mut parents := []&autograd.Variable[T]{cap: expected_len}
	for index, arg in args {
		match arg {
			autograd.Variable[T] {
				parents << arg
			}
			else {
				return error('LayerNormGate.cache: argument ${index} must be a Variable')
			}
		}
	}
	result.grad = vtl.zeros_like[T](result.value)
	result.requires_grad = true
	autograd.register[T]('LayerNorm', voidptr(g), layer_norm_gate_backward_dispatch[T], result,
		parents)!
}
