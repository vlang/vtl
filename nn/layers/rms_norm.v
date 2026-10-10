module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types

// RMSNormConfig controls numerical stability and the per-element weight.
@[params]
pub struct RMSNormConfig {
pub:
	eps                ?f64
	elementwise_affine bool = true
}

// RMSNormLayer normalizes each trailing block by its root mean square.
pub struct RMSNormLayer[T] {
pub:
	normalized_shape []int
	eps              f64
pub mut:
	weight &autograd.Variable[T] = unsafe { nil }
}

// rms_norm_layer creates an RMSNorm layer. normalized_shape must match the
// trailing dimensions of each input handled by the layer.
pub fn rms_norm_layer[T](ctx &autograd.Context[T], normalized_shape []int, config RMSNormConfig) types.Layer[T] {
	weight := if config.elementwise_affine {
		ctx.variable(vtl.ones[T](normalized_shape))
	} else {
		unsafe { &autograd.Variable[T](nil) }
	}
	layer := &RMSNormLayer[T]{
		normalized_shape: normalized_shape.clone()
		eps:              config.eps or { rms_norm_default_eps[T]() }
		weight:           weight
	}
	return types.layer[T](voidptr(layer), rms_norm_layer_output_shape_dispatch[T],
		rms_norm_layer_variables_dispatch[T], rms_norm_layer_forward_dispatch[T])
}

fn rms_norm_default_eps[T]() f64 {
	$if T is f32 {
		return 1.1920928955078125e-7
	} $else {
		return 2.220446049250313e-16
	}
}

// output_shape returns the normalized feature shape.
pub fn (layer &RMSNormLayer[T]) output_shape() []int {
	return layer.normalized_shape.clone()
}

// variables returns the learnable weight, when elementwise_affine is enabled.
pub fn (layer &RMSNormLayer[T]) variables() []&autograd.Variable[T] {
	if layer.weight != unsafe { nil } {
		return [layer.weight]
	}
	return []&autograd.Variable[T]{}
}

// forward normalizes trailing dimensions and records autograd when requested.
pub fn (layer &RMSNormLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	weight_value := if layer.weight != unsafe { nil } {
		layer.weight.value
	} else {
		unsafe { &vtl.Tensor[T](nil) }
	}
	output := internal.rms_norm_forward[T](input.value, weight_value, layer.normalized_shape,
		layer.eps)!
	mut result := input.context.variable(output,
		requires_grad: input.requires_grad || (layer.weight != unsafe { nil } && layer.weight.requires_grad)
	)
	if result.requires_grad {
		gate := rms_norm_gate[T](input.value, weight_value, layer.normalized_shape, layer.eps)
		if layer.weight != unsafe { nil } {
			gate.cache(mut result, input, layer.weight)!
		} else {
			gate.cache(mut result, input)!
		}
	}
	return result
}

fn rms_norm_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&RMSNormLayer[T](layer)).output_shape() }
}

fn rms_norm_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&RMSNormLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn rms_norm_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&RMSNormLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

// RMSNormGate retains the input and optional weight for reverse-mode gradients.
pub struct RMSNormGate[T] {
	input            &vtl.Tensor[T] = unsafe { nil }
	weight           &vtl.Tensor[T] = unsafe { nil }
	normalized_shape []int
	eps              f64
}

// rms_norm_gate creates an RMSNorm backward gate.
pub fn rms_norm_gate[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], normalized_shape []int, eps f64) &RMSNormGate[T] {
	return &RMSNormGate[T]{
		input:            input
		weight:           weight
		normalized_shape: normalized_shape.clone()
		eps:              eps
	}
}

// backward returns input and optional per-element weight gradients.
pub fn (gate &RMSNormGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return internal.rms_norm_backward[T](payload.variable.grad, gate.input, gate.weight,
		gate.normalized_shape, gate.eps)
}

fn rms_norm_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&RMSNormGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers the RMSNorm gate with its input and optional weight.
pub fn (gate &RMSNormGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	expected_len := if gate.weight != unsafe { nil } { 2 } else { 1 }
	if args.len != expected_len {
		return error('RMSNormGate.cache: expected ${expected_len} variable arguments')
	}
	mut parents := []&autograd.Variable[T]{cap: expected_len}
	for index, arg in args {
		match arg {
			autograd.Variable[T] { parents << arg }
			else { return error('RMSNormGate.cache: argument ${index} must be a Variable') }
		}
	}
	result.grad = vtl.zeros_like[T](result.value)
	result.requires_grad = true
	autograd.register[T]('RMSNorm', voidptr(gate), rms_norm_gate_backward_dispatch[T], result,
		parents)!
}
