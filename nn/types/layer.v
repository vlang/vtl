module types

import vtl.autograd

// LayerForwardFn forwards a layer using opaque wrapper pointers.
pub type LayerForwardFn = fn (layer voidptr, input voidptr) !voidptr

// LayerOutputShapeFn returns a layer output shape from an opaque wrapper pointer.
pub type LayerOutputShapeFn = fn (layer voidptr) []int

// LayerVariablesFn returns trainable variable pointers from an opaque wrapper pointer.
pub type LayerVariablesFn = fn (layer voidptr) []voidptr

// Layer is an opaque wrapper for a neural network layer.
pub struct Layer[T] {
	ptr             voidptr
	output_shape_fn LayerOutputShapeFn = unsafe { nil }
	variables_fn    LayerVariablesFn   = unsafe { nil }
	forward_fn      LayerForwardFn     = unsafe { nil }
}

// layer creates a typed wrapper around a concrete neural network layer.
pub fn layer[T](ptr voidptr, output_shape_fn LayerOutputShapeFn, variables_fn LayerVariablesFn, forward_fn LayerForwardFn) Layer[T] {
	return Layer[T]{
		ptr:             ptr
		output_shape_fn: output_shape_fn
		variables_fn:    variables_fn
		forward_fn:      forward_fn
	}
}

// variable_ptrs_to_voidptrs converts typed variable pointers at wrapper edges.
pub fn variable_ptrs_to_voidptrs[T](vars []&autograd.Variable[T]) []voidptr {
	mut result := []voidptr{cap: vars.len}
	for variable in vars {
		result << voidptr(variable)
	}
	return result
}

// output_shape returns the layer output shape.
pub fn (layer Layer[T]) output_shape() []int {
	return layer.output_shape_fn(layer.ptr)
}

// variables returns the trainable variables owned by the layer.
pub fn (layer Layer[T]) variables() []&autograd.Variable[T] {
	ptrs := layer.variables_fn(layer.ptr)
	mut vars := []&autograd.Variable[T]{cap: ptrs.len}
	for ptr in ptrs {
		vars << unsafe { &autograd.Variable[T](ptr) }
	}
	return vars
}

// forward runs a forward pass through the wrapped layer.
pub fn (layer Layer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	result := layer.forward_fn(layer.ptr, voidptr(input))!
	return unsafe { &autograd.Variable[T](result) }
}
