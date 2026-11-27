module layers

import vtl.autograd
import vtl
import vtl.nn.internal
import vtl.nn.gates.layers
import vtl.nn.types

// MaxPool2DLayer is a layer that implements the maxpooling operation.
pub struct MaxPool2DLayer[T] {
	input_shape []int
	kernel      []int
	padding     []int
	stride      []int
}

// maxpool2d_layer exposes this operation as part of the public API.
pub fn maxpool2d_layer[T](ctx &autograd.Context[T], input_shape []int, kernel []int, padding []int, stride []int) types.Layer[T] {
	layer := &MaxPool2DLayer[T]{
		input_shape: input_shape.clone()
		kernel:      kernel.clone()
		padding:     padding.clone()
		stride:      stride.clone()
	}
	return types.layer[T](voidptr(layer), max_pool2_d_layer_output_shape_dispatch[T],
		max_pool2_d_layer_variables_dispatch[T], max_pool2_d_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &MaxPool2DLayer[T]) output_shape() []int {
	c := layer.input_shape[0]
	h := layer.input_shape[1]
	w := layer.input_shape[2]
	kh := layer.kernel[0]
	kw := layer.kernel[1]
	ph := layer.padding[0]
	pw := layer.padding[1]
	sh := layer.stride[0]
	sw := layer.stride[1]

	return [c, (h - kh + 2 * ph) / sh + 1, (w - kw + 2 * pw) / sw + 1]
}

// variables exposes this operation as part of the public API.
pub fn (layer &MaxPool2DLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &MaxPool2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	max_indices, output := internal.maxpool2d[T](input.value, layer.kernel, layer.padding,
		layer.stride)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := layers.maxpool2d_gate_with_input[T](max_indices, input.value, layer.kernel,
			input.value.shape, layer.stride, layer.padding)
		gate.cache(mut result, input, layer.kernel, layer.padding, layer.stride)!
	}

	return result
}

fn max_pool2_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&MaxPool2DLayer[T](layer)).output_shape() }
}

fn max_pool2_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&MaxPool2DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn max_pool2_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&MaxPool2DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

pub struct MaxPool3DLayer[T] {
	input_shape []int
	kernel      []int
	padding     []int
	stride      []int
}

pub fn maxpool3d_layer[T](ctx &autograd.Context[T], input_shape []int, kernel []int, padding []int, stride []int) types.Layer[T] {
	layer := &MaxPool3DLayer[T]{
		input_shape: input_shape.clone()
		kernel:      kernel.clone()
		padding:     padding.clone()
		stride:      stride.clone()
	}
	return types.layer[T](voidptr(layer), max_pool3_d_layer_output_shape_dispatch[T],
		max_pool3_d_layer_variables_dispatch[T], max_pool3_d_layer_forward_dispatch[T])
}

pub fn (layer &MaxPool3DLayer[T]) output_shape() []int {
	return [
		layer.input_shape[0],
		(layer.input_shape[1] - layer.kernel[0] + 2 * layer.padding[0]) / layer.stride[0] + 1,
		(layer.input_shape[2] - layer.kernel[1] + 2 * layer.padding[1]) / layer.stride[1] + 1,
		(layer.input_shape[3] - layer.kernel[2] + 2 * layer.padding[2]) / layer.stride[2] + 1,
	]
}

pub fn (layer &MaxPool3DLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &MaxPool3DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	max_indices, output := internal.maxpool3d[T](input.value, layer.kernel, layer.padding, layer.stride)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := maxpool3d_gate[T](max_indices, input.value.shape, layer.kernel, layer.stride)
		gate.cache(mut result, input)!
	}
	return result
}

fn max_pool3_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&MaxPool3DLayer[T](layer)).output_shape() }
}

fn max_pool3_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&MaxPool3DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn max_pool3_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&MaxPool3DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

pub struct MaxPool3DGate[T] {
	max_indices &vtl.Tensor[int] = unsafe { nil }
	shape       []int
}

pub fn maxpool3d_gate[T](max_indices &vtl.Tensor[int], shape []int, kernel []int, stride []int) &MaxPool3DGate[T] {
	return &MaxPool3DGate[T]{ max_indices: max_indices, shape: shape }
}

pub fn (g &MaxPool3DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return [internal.maxpool3d_backward[T](g.shape, g.max_indices, payload.variable.grad)!]
}

fn max_pool3_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&MaxPool3DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &MaxPool3DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('MaxPool3D', voidptr(g), max_pool3_d_gate_backward_dispatch[T],
				result, [a])!
		}
		else {}
	}
}
