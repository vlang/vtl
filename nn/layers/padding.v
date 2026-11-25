module layers

import vtl
import vtl.autograd
import vtl.nn.types

// Pad2DLayer pads channel-first image tensors [batch, channels, height, width].
// padding is [left, right, top, bottom].
pub struct Pad2DLayer[T] {
	padding     []int
	mode        vtl.PadMode
	constant    T
	input_shape []int
}

pub fn pad2d_layer[T](ctx &autograd.Context[T], input_shape []int, padding []int, mode vtl.PadMode, constant T) types.Layer[T] {
	layer := &Pad2DLayer[T]{
		padding:     padding.clone()
		mode:        mode
		constant:    constant
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), pad_2_d_layer_output_shape_dispatch[T],
		pad_2_d_layer_variables_dispatch[T], pad_2_d_layer_forward_dispatch[T])
}

pub fn (layer &Pad2DLayer[T]) output_shape() []int {
	return [
		layer.input_shape[0],
		layer.input_shape[1] + layer.padding[2] + layer.padding[3],
		layer.input_shape[2] + layer.padding[0] + layer.padding[1],
	]
}

pub fn (layer &Pad2DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

pub fn (layer &Pad2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	pad_width := [[0, 0], [0, 0], [layer.padding[2], layer.padding[3]],
		[layer.padding[0], layer.padding[1]]]
	return input.pad(pad_width, layer.mode, layer.constant)
}

fn pad_2_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&Pad2DLayer[T](layer)).output_shape() }
}

fn pad_2_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&Pad2DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn pad_2_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&Pad2DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

// Pad1DLayer pads channel-first sequence tensors [batch, channels, length].
// padding is [left, right].
pub struct Pad1DLayer[T] {
	padding     []int
	mode        vtl.PadMode
	constant    T
	input_shape []int
}

pub fn pad1d_layer[T](ctx &autograd.Context[T], input_shape []int, padding []int, mode vtl.PadMode, constant T) types.Layer[T] {
	layer := &Pad1DLayer[T]{
		padding:     padding.clone()
		mode:        mode
		constant:    constant
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), pad_1_d_layer_output_shape_dispatch[T],
		pad_1_d_layer_variables_dispatch[T], pad_1_d_layer_forward_dispatch[T])
}

pub fn (layer &Pad1DLayer[T]) output_shape() []int {
	return [layer.input_shape[0], layer.input_shape[1] + layer.padding[0] + layer.padding[1]]
}

pub fn (layer &Pad1DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

pub fn (layer &Pad1DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	pad_width := [[0, 0], [0, 0], [layer.padding[0], layer.padding[1]]]
	return input.pad(pad_width, layer.mode, layer.constant)
}

fn pad_1_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&Pad1DLayer[T](layer)).output_shape() }
}

fn pad_1_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&Pad1DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn pad_1_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&Pad1DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

// Pad3DLayer pads channel-first volumetric tensors [batch, channels, depth, height, width].
// padding is [left, right, top, bottom, front, back].
pub struct Pad3DLayer[T] {
	padding     []int
	mode        vtl.PadMode
	constant    T
	input_shape []int
}

pub fn pad3d_layer[T](ctx &autograd.Context[T], input_shape []int, padding []int, mode vtl.PadMode, constant T) types.Layer[T] {
	layer := &Pad3DLayer[T]{
		padding:     padding.clone()
		mode:        mode
		constant:    constant
		input_shape: input_shape.clone()
	}
	return types.layer[T](voidptr(layer), pad_3_d_layer_output_shape_dispatch[T],
		pad_3_d_layer_variables_dispatch[T], pad_3_d_layer_forward_dispatch[T])
}

pub fn (layer &Pad3DLayer[T]) output_shape() []int {
	return [
		layer.input_shape[0],
		layer.input_shape[1] + layer.padding[4] + layer.padding[5],
		layer.input_shape[2] + layer.padding[2] + layer.padding[3],
		layer.input_shape[3] + layer.padding[0] + layer.padding[1],
	]
}

pub fn (layer &Pad3DLayer[T]) variables() []&autograd.Variable[T] {
	return []
}

pub fn (layer &Pad3DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	pad_width := [
		[0, 0],
		[0, 0],
		[layer.padding[4], layer.padding[5]],
		[layer.padding[2], layer.padding[3]],
		[layer.padding[0], layer.padding[1]],
	]
	return input.pad(pad_width, layer.mode, layer.constant)
}

fn pad_3_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&Pad3DLayer[T](layer)).output_shape() }
}

fn pad_3_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&Pad3DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn pad_3_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&Pad3DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
