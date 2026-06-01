module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// ReLULayer is a layer that applies the rectified linear unit function element-wise.
pub struct ReLULayer[T] {
	output_shape []int
}

// relu_layer exposes this operation as part of the public API.
pub fn relu_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	layer := &ReLULayer[T]{
		output_shape: output_shape.clone()
	}
	return types.layer[T](voidptr(layer), re_lu_layer_output_shape_dispatch[T],
		re_lu_layer_variables_dispatch[T], re_lu_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &ReLULayer[T]) output_shape() []int {
	return layer.output_shape
}

// variables exposes this operation as part of the public API.
pub fn (_ &ReLULayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &ReLULayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	mut output := &vtl.Tensor[T](unsafe { nil })
	$if sizeof(T) == 4 {
		out_f32 := relu_forward_f32(unsafe { &vtl.Tensor[f32](input.value) })!
		output = unsafe { &vtl.Tensor[T](out_f32) }
	} $else {
		output = internal.relu[T](input.value)
	}
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.relu_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}

fn re_lu_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&ReLULayer[T](layer)).output_shape() }
}

fn re_lu_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&ReLULayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn re_lu_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&ReLULayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
