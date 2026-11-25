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
	grad := internal.avgpool2d_backward[T](payload.variable.grad, g.input, g.kernel, g.padding,
		g.stride)!
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
			autograd.register[T]('GlobalAvgPool2D', voidptr(g),
				global_avg_pool2_d_gate_backward_dispatch[T], result, [a])!
		}
		else {}
	}
}

// AdaptiveAvgPool2DLayer produces a fixed spatial output shape from a 4D
// channel-first input, matching torch.nn.AdaptiveAvgPool2d for f32/f64 CPU tensors.
pub struct AdaptiveAvgPool2DLayer[T] {
	output_size []int
	input_shape []int
}

// adaptive_avgpool2d_layer creates an AdaptiveAvgPool2D layer.
pub fn adaptive_avgpool2d_layer[T](ctx &autograd.Context[T], input_shape []int, output_size []int) types.Layer[T] {
	layer := &AdaptiveAvgPool2DLayer[T]{
		output_size: output_size.clone()
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), adaptive_avg_pool2_d_layer_output_shape_dispatch[T],
		adaptive_avg_pool2_d_layer_variables_dispatch[T],
		adaptive_avg_pool2_d_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &AdaptiveAvgPool2DLayer[T]) output_shape() []int {
	return [layer.input_shape[0], layer.output_size[0], layer.output_size[1]]
}

// variables exposes this operation as part of the public API.
pub fn (layer &AdaptiveAvgPool2DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

// forward exposes this operation as part of the public API.
pub fn (layer &AdaptiveAvgPool2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.adaptive_avgpool2d_forward[T](input.value, layer.output_size)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := adaptive_avgpool2d_gate[T](input.value, layer.output_size)
		gate.cache(mut result, input)!
	}
	return result
}

fn adaptive_avg_pool2_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&AdaptiveAvgPool2DLayer[T](layer)).output_shape() }
}

fn adaptive_avg_pool2_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&AdaptiveAvgPool2DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn adaptive_avg_pool2_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&AdaptiveAvgPool2DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

// AdaptiveAvgPool2DGate stores the input needed for the backward pass.
pub struct AdaptiveAvgPool2DGate[T] {
	input       &vtl.Tensor[T] = unsafe { nil }
	output_size []int
}

// adaptive_avgpool2d_gate creates an autograd gate for adaptive average pooling.
pub fn adaptive_avgpool2d_gate[T](input &vtl.Tensor[T], output_size []int) &AdaptiveAvgPool2DGate[T] {
	return &AdaptiveAvgPool2DGate[T]{
		input:       input
		output_size: output_size.clone()
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &AdaptiveAvgPool2DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := internal.adaptive_avgpool2d_backward[T](payload.variable.grad, g.input, g.output_size)!
	return [grad]
}

fn adaptive_avg_pool2_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&AdaptiveAvgPool2DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &AdaptiveAvgPool2DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('AdaptiveAvgPool2D', voidptr(g),
				adaptive_avg_pool2_d_gate_backward_dispatch[T], result, [a])!
		}
		else {}
	}
}

pub struct AdaptiveMaxPool2DLayer[T] {
	output_size []int
	input_shape []int
}

pub fn adaptive_maxpool2d_layer[T](ctx &autograd.Context[T], input_shape []int, output_size []int) types.Layer[T] {
	layer := &AdaptiveMaxPool2DLayer[T]{
		output_size: output_size.clone()
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), adaptive_max_pool2_d_layer_output_shape_dispatch[T],
		adaptive_max_pool2_d_layer_variables_dispatch[T],
		adaptive_max_pool2_d_layer_forward_dispatch[T])
}

pub fn (layer &AdaptiveMaxPool2DLayer[T]) output_shape() []int {
	return [layer.input_shape[0], layer.output_size[0], layer.output_size[1]]
}

pub fn (layer &AdaptiveMaxPool2DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

pub fn (layer &AdaptiveMaxPool2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.adaptive_maxpool2d_forward[T](input.value, layer.output_size)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := adaptive_maxpool2d_gate[T](input.value, layer.output_size)
		gate.cache(mut result, input)!
	}
	return result
}

fn adaptive_max_pool2_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&AdaptiveMaxPool2DLayer[T](layer)).output_shape() }
}

fn adaptive_max_pool2_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&AdaptiveMaxPool2DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn adaptive_max_pool2_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&AdaptiveMaxPool2DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

pub struct AdaptiveMaxPool2DGate[T] {
	input       &vtl.Tensor[T] = unsafe { nil }
	output_size []int
}

pub fn adaptive_maxpool2d_gate[T](input &vtl.Tensor[T], output_size []int) &AdaptiveMaxPool2DGate[T] {
	return &AdaptiveMaxPool2DGate[T]{
		input:       input
		output_size: output_size.clone()
	}
}

pub fn (g &AdaptiveMaxPool2DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := internal.adaptive_maxpool2d_backward[T](payload.variable.grad, g.input, g.output_size)!
	return [grad]
}

fn adaptive_max_pool2_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&AdaptiveMaxPool2DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &AdaptiveMaxPool2DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('AdaptiveMaxPool2D', voidptr(g),
				adaptive_max_pool2_d_gate_backward_dispatch[T], result, [a])!
		}
		else {}
	}
}

pub struct AdaptiveAvgPool1DLayer[T] {
	output_size int
	input_shape []int
}

pub fn adaptive_avgpool1d_layer[T](ctx &autograd.Context[T], input_shape []int, output_size int) types.Layer[T] {
	layer := &AdaptiveAvgPool1DLayer[T]{
		output_size: output_size
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), adaptive_avg_pool1_d_layer_output_shape_dispatch[T],
		adaptive_avg_pool1_d_layer_variables_dispatch[T],
		adaptive_avg_pool1_d_layer_forward_dispatch[T])
}

pub fn (layer &AdaptiveAvgPool1DLayer[T]) output_shape() []int {
	return [layer.input_shape[0], layer.output_size]
}

pub fn (layer &AdaptiveAvgPool1DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

pub fn (layer &AdaptiveAvgPool1DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.adaptive_avgpool1d_forward[T](input.value, layer.output_size)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := adaptive_avgpool1d_gate[T](input.value, layer.output_size)
		gate.cache(mut result, input)!
	}
	return result
}

fn adaptive_avg_pool1_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&AdaptiveAvgPool1DLayer[T](layer)).output_shape() }
}

fn adaptive_avg_pool1_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&AdaptiveAvgPool1DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn adaptive_avg_pool1_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&AdaptiveAvgPool1DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

pub struct AdaptiveAvgPool1DGate[T] {
	input       &vtl.Tensor[T] = unsafe { nil }
	output_size int
}

pub fn adaptive_avgpool1d_gate[T](input &vtl.Tensor[T], output_size int) &AdaptiveAvgPool1DGate[T] {
	return &AdaptiveAvgPool1DGate[T]{
		input:       input
		output_size: output_size
	}
}

pub fn (g &AdaptiveAvgPool1DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := internal.adaptive_avgpool1d_backward[T](payload.variable.grad, g.input, g.output_size)!
	return [grad]
}

fn adaptive_avg_pool1_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&AdaptiveAvgPool1DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &AdaptiveAvgPool1DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('AdaptiveAvgPool1D', voidptr(g),
				adaptive_avg_pool1_d_gate_backward_dispatch[T], result, [a])!
		}
		else {}
	}
}

pub struct AdaptiveMaxPool1DLayer[T] {
	output_size int
	input_shape []int
}

pub fn adaptive_maxpool1d_layer[T](ctx &autograd.Context[T], input_shape []int, output_size int) types.Layer[T] {
	layer := &AdaptiveMaxPool1DLayer[T]{
		output_size: output_size
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), adaptive_max_pool1_d_layer_output_shape_dispatch[T],
		adaptive_max_pool1_d_layer_variables_dispatch[T],
		adaptive_max_pool1_d_layer_forward_dispatch[T])
}

pub fn (layer &AdaptiveMaxPool1DLayer[T]) output_shape() []int {
	return [layer.input_shape[0], layer.output_size]
}

pub fn (layer &AdaptiveMaxPool1DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

pub fn (layer &AdaptiveMaxPool1DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.adaptive_maxpool1d_forward[T](input.value, layer.output_size)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := adaptive_maxpool1d_gate[T](input.value, layer.output_size)
		gate.cache(mut result, input)!
	}
	return result
}

fn adaptive_max_pool1_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&AdaptiveMaxPool1DLayer[T](layer)).output_shape() }
}

fn adaptive_max_pool1_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&AdaptiveMaxPool1DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn adaptive_max_pool1_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&AdaptiveMaxPool1DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

pub struct AdaptiveMaxPool1DGate[T] {
	input       &vtl.Tensor[T] = unsafe { nil }
	output_size int
}

pub fn adaptive_maxpool1d_gate[T](input &vtl.Tensor[T], output_size int) &AdaptiveMaxPool1DGate[T] {
	return &AdaptiveMaxPool1DGate[T]{
		input:       input
		output_size: output_size
	}
}

pub fn (g &AdaptiveMaxPool1DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := internal.adaptive_maxpool1d_backward[T](payload.variable.grad, g.input, g.output_size)!
	return [grad]
}

fn adaptive_max_pool1_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&AdaptiveMaxPool1DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &AdaptiveMaxPool1DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('AdaptiveMaxPool1D', voidptr(g),
				adaptive_max_pool1_d_gate_backward_dispatch[T], result, [a])!
		}
		else {}
	}
}

pub struct AdaptiveAvgPool3DLayer[T] {
	output_size []int
	input_shape []int
}

pub fn adaptive_avgpool3d_layer[T](ctx &autograd.Context[T], input_shape []int, output_size []int) types.Layer[T] {
	layer := &AdaptiveAvgPool3DLayer[T]{
		output_size: output_size.clone()
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), adaptive_avg_pool3_d_layer_output_shape_dispatch[T],
		adaptive_avg_pool3_d_layer_variables_dispatch[T],
		adaptive_avg_pool3_d_layer_forward_dispatch[T])
}

pub fn (layer &AdaptiveAvgPool3DLayer[T]) output_shape() []int {
	return [layer.input_shape[0], layer.output_size[0], layer.output_size[1], layer.output_size[2]]
}

pub fn (layer &AdaptiveAvgPool3DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

pub fn (layer &AdaptiveAvgPool3DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.adaptive_avgpool3d_forward[T](input.value, layer.output_size)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := adaptive_avgpool3d_gate[T](input.value, layer.output_size)
		gate.cache(mut result, input)!
	}
	return result
}

fn adaptive_avg_pool3_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&AdaptiveAvgPool3DLayer[T](layer)).output_shape() }
}

fn adaptive_avg_pool3_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&AdaptiveAvgPool3DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn adaptive_avg_pool3_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&AdaptiveAvgPool3DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

pub struct AdaptiveAvgPool3DGate[T] {
	input       &vtl.Tensor[T] = unsafe { nil }
	output_size []int
}

pub fn adaptive_avgpool3d_gate[T](input &vtl.Tensor[T], output_size []int) &AdaptiveAvgPool3DGate[T] {
	return &AdaptiveAvgPool3DGate[T]{
		input:       input
		output_size: output_size.clone()
	}
}

pub fn (g &AdaptiveAvgPool3DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := internal.adaptive_avgpool3d_backward[T](payload.variable.grad, g.input, g.output_size)!
	return [grad]
}

fn adaptive_avg_pool3_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&AdaptiveAvgPool3DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &AdaptiveAvgPool3DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('AdaptiveAvgPool3D', voidptr(g),
				adaptive_avg_pool3_d_gate_backward_dispatch[T], result, [a])!
		}
		else {}
	}
}

pub struct AdaptiveMaxPool3DLayer[T] {
	output_size []int
	input_shape []int
}

pub fn adaptive_maxpool3d_layer[T](ctx &autograd.Context[T], input_shape []int, output_size []int) types.Layer[T] {
	layer := &AdaptiveMaxPool3DLayer[T]{
		output_size: output_size.clone()
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), adaptive_max_pool3_d_layer_output_shape_dispatch[T],
		adaptive_max_pool3_d_layer_variables_dispatch[T],
		adaptive_max_pool3_d_layer_forward_dispatch[T])
}

pub fn (layer &AdaptiveMaxPool3DLayer[T]) output_shape() []int {
	return [layer.input_shape[0], layer.output_size[0], layer.output_size[1], layer.output_size[2]]
}

pub fn (layer &AdaptiveMaxPool3DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

pub fn (layer &AdaptiveMaxPool3DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.adaptive_maxpool3d_forward[T](input.value, layer.output_size)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := adaptive_maxpool3d_gate[T](input.value, layer.output_size)
		gate.cache(mut result, input)!
	}
	return result
}

fn adaptive_max_pool3_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&AdaptiveMaxPool3DLayer[T](layer)).output_shape() }
}

fn adaptive_max_pool3_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&AdaptiveMaxPool3DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn adaptive_max_pool3_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&AdaptiveMaxPool3DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

pub struct AdaptiveMaxPool3DGate[T] {
	input       &vtl.Tensor[T] = unsafe { nil }
	output_size []int
}

pub fn adaptive_maxpool3d_gate[T](input &vtl.Tensor[T], output_size []int) &AdaptiveMaxPool3DGate[T] {
	return &AdaptiveMaxPool3DGate[T]{
		input:       input
		output_size: output_size.clone()
	}
}

pub fn (g &AdaptiveMaxPool3DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := internal.adaptive_maxpool3d_backward[T](payload.variable.grad, g.input, g.output_size)!
	return [grad]
}

fn adaptive_max_pool3_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&AdaptiveMaxPool3DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &AdaptiveMaxPool3DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('AdaptiveMaxPool3D', voidptr(g),
				adaptive_max_pool3_d_gate_backward_dispatch[T], result, [a])!
		}
		else {}
	}
}

pub struct AveragePool1DLayer[T] {
	kernel      int
	padding     int
	stride      int
	input_shape []int
}

pub fn avgpool1d_layer[T](ctx &autograd.Context[T], input_shape []int, kernel int, padding int, stride int) types.Layer[T] {
	layer := &AveragePool1DLayer[T]{
		kernel:      kernel
		padding:     padding
		stride:      stride
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), average_pool1_d_layer_output_shape_dispatch[T],
		average_pool1_d_layer_variables_dispatch[T],
		average_pool1_d_layer_forward_dispatch[T])
}

pub fn (layer &AveragePool1DLayer[T]) output_shape() []int {
	return [layer.input_shape[0],
		(layer.input_shape[1] - layer.kernel + 2 * layer.padding) / layer.stride + 1]
}

pub fn (layer &AveragePool1DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

pub fn (layer &AveragePool1DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.avgpool1d_forward[T](input.value, layer.kernel, layer.padding, layer.stride)!
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := avgpool1d_gate[T](input.value, layer.kernel, layer.padding, layer.stride)
		gate.cache(mut result, input)!
	}
	return result
}

fn average_pool1_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&AveragePool1DLayer[T](layer)).output_shape() }
}

fn average_pool1_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&AveragePool1DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn average_pool1_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&AveragePool1DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

pub struct AvgPool1DGate[T] {
	input   &vtl.Tensor[T] = unsafe { nil }
	kernel  int
	padding int
	stride  int
}

pub fn avgpool1d_gate[T](input &vtl.Tensor[T], kernel int, padding int, stride int) &AvgPool1DGate[T] {
	return &AvgPool1DGate[T]{
		input:   input
		kernel:  kernel
		padding: padding
		stride:  stride
	}
}

pub fn (g &AvgPool1DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := internal.avgpool1d_backward[T](payload.variable.grad, g.input, g.kernel, g.padding, g.stride)!
	return [grad]
}

fn avg_pool1_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&AvgPool1DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &AvgPool1DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('AvgPool1D', voidptr(g), avg_pool1_d_gate_backward_dispatch[T],
				result, [a])!
		}
		else {}
	}
}
