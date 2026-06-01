module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types

// AveragePool2DLayer applies 2D average pooling over a 4D input.
//
// Input:    `[batch, channels, H, W]`
// Output:   `[batch, channels, out_H, out_W]`
//
// Config options:
//   - `kernel` — pool window size in (H, W) (default: determined by `input_shape`)
//   - `padding` — zero-border padding before pooling (default: [0,0])
//   - `stride`  — pool stride in (H, W) (default: same as kernel)
pub struct AveragePool2DLayer[T] {
	kernel      []int
	padding     []int
	stride      []int
	input_shape []int
}

// avgpool2d_layer creates an AveragePool2DLayer.
pub fn avgpool2d_layer[T](ctx &autograd.Context[T], input_shape []int, kernel []int, padding []int, stride []int) types.Layer[T] {
	layer := &AveragePool2DLayer[T]{
		kernel:      kernel.clone()
		padding:     padding.clone()
		stride:      stride.clone()
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), average_pool2_d_layer_output_shape_dispatch[T],
		average_pool2_d_layer_variables_dispatch[T], average_pool2_d_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &AveragePool2DLayer[T]) output_shape() []int {
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
pub fn (layer &AveragePool2DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

// forward exposes this operation as part of the public API.
pub fn (layer &AveragePool2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.avgpool2d_forward[T](input.value, layer.kernel, layer.padding, layer.stride)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := avgpool2d_gate[T](input.value, layer.kernel, layer.padding, layer.stride)
		gate.cache(mut result, input)!
	}
	return result
}

fn average_pool2_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&AveragePool2DLayer[T](layer)).output_shape() }
}

fn average_pool2_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&AveragePool2DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn average_pool2_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&AveragePool2DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

// AvgPool2DGate defines a public data structure for this module.
pub struct AvgPool2DGate[T] {
	input   &vtl.Tensor[T] = unsafe { nil }
	kernel  []int
	padding []int
	stride  []int
}

// avgpool2d_gate exposes this operation as part of the public API.
pub fn avgpool2d_gate[T](input &vtl.Tensor[T], kernel []int, padding []int, stride []int) &AvgPool2DGate[T] {
	return &AvgPool2DGate[T]{
		input:   input
		kernel:  kernel
		padding: padding
		stride:  stride
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &AvgPool2DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := internal.avgpool2d_backward[T](payload.variable.grad, g.kernel, g.padding, g.stride)!
	return [grad]
}

fn avg_pool2_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&AvgPool2DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &AvgPool2DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('AvgPool2D', voidptr(g), avg_pool2_d_gate_backward_dispatch[T],
				result, [a])!
		}
		else {}
	}
}

// GlobalAvgPool2DLayer applies global average pooling over spatial dimensions.
//
// Input:    `[batch, channels, H, W]`
// Output:   `[batch, channels, 1, 1]` — one value per channel per sample
pub struct GlobalAvgPool2DLayer[T] {}

// global_avgpool2d_layer creates a GlobalAvgPool2DLayer.
pub fn global_avgpool2d_layer[T](ctx &autograd.Context[T]) types.Layer[T] {
	layer := &GlobalAvgPool2DLayer[T]{}
	return types.layer[T](voidptr(layer), global_avg_pool2_d_layer_output_shape_dispatch[T],
		global_avg_pool2_d_layer_variables_dispatch[T],
		global_avg_pool2_d_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &GlobalAvgPool2DLayer[T]) output_shape() []int {
	return [-1, -1]
}

// variables exposes this operation as part of the public API.
pub fn (layer &GlobalAvgPool2DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

// forward exposes this operation as part of the public API.
pub fn (layer &GlobalAvgPool2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.global_avgpool2d_forward[T](input.value)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := global_avgpool2d_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}

fn global_avg_pool2_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&GlobalAvgPool2DLayer[T](layer)).output_shape() }
}

fn global_avg_pool2_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&GlobalAvgPool2DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn global_avg_pool2_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&GlobalAvgPool2DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

// GlobalAvgPool2DGate defines a public data structure for this module.
pub struct GlobalAvgPool2DGate[T] {
	input &vtl.Tensor[T] = unsafe { nil }
}

// global_avgpool2d_gate exposes this operation as part of the public API.
pub fn global_avgpool2d_gate[T](input &vtl.Tensor[T]) &GlobalAvgPool2DGate[T] {
	return &GlobalAvgPool2DGate[T]{
		input: input
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &GlobalAvgPool2DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := internal.global_avgpool2d_backward[T](payload.variable.grad, g.input)!
	return [grad]
}

fn global_avg_pool2_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&GlobalAvgPool2DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &GlobalAvgPool2DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('GlobalAvgPool2D', voidptr(g),
				global_avg_pool2_d_gate_backward_dispatch[T], result, [a])!
		}
		else {}
	}
}
