module layers

import vtl
import vtl.la
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers
import vtl.nn.types

// LinearLayer applies a linear transformation: y = x·Wᵀ + b
//
// Input:  [..., in_features]
// Output: [..., out_features]
//
// Weights shape: `[out_features, in_features]`
// Bias shape:    `[1, out_features]`
pub struct LinearLayer[T] {
pub:
	weights &autograd.Variable[T] = unsafe { nil }
	bias    &autograd.Variable[T] = unsafe { nil }
}

// linear_layer creates a LinearLayer.
pub fn linear_layer[T](ctx &autograd.Context[T], input_dim int, output_dim int) types.Layer[T] {
	weights := internal.kaiming_normal[T]([output_dim, input_dim])
	bias := vtl.zeros[T]([1, output_dim])
	return types.Layer[T](&LinearLayer[T]{
		weights: ctx.variable(weights)
		bias:    ctx.variable(bias)
	})
}

pub fn (layer &LinearLayer[T]) output_shape() []int {
	return [layer.weights.value.shape[0]]
}

pub fn (layer &LinearLayer[T]) variables() []&autograd.Variable[T] {
	return [layer.weights, layer.bias]
}

pub fn (layer &LinearLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	mut output := &vtl.Tensor[T](unsafe { nil })
	$if sizeof(T) == 8 {
		// GPU activation chain input is passed via session; take-from-variable needs mut input (follow-up).
		in_gpu := unsafe { nil }
		out := linear_forward_f64(unsafe { &vtl.Tensor[f64](input.value) },
			unsafe { &vtl.Tensor[f64](layer.weights.value) },
			unsafe { &vtl.Tensor[f64](layer.bias.value) }, in_gpu, input.context.device_session)!
		output = unsafe { &vtl.Tensor[T](out) }
	} $else $if sizeof(T) == 4 {
		out_f32 := linear_forward_f32(unsafe { &vtl.Tensor[f32](input.value) },
			unsafe { &vtl.Tensor[f32](layer.weights.value) },
			unsafe { &vtl.Tensor[f32](layer.bias.value) })!
		output = unsafe { &vtl.Tensor[T](out_f32) }
	} $else {
		output = la.matmul[T](input.value, layer.weights.value.t()!)!.add[T](layer.bias.value)!
	}
	mut result := input.context.variable(output)

	$if sizeof(T) == 8 {
		linear_bind_output_gpu(voidptr(result))
	}

	if input.requires_grad || layer.weights.requires_grad || layer.bias.requires_grad {
		gate := layers.linear_gate[T](input, layer.weights, layer.bias)
		gate.cache(mut result, input, layer.weights, layer.bias)!
	}

	return result
}
