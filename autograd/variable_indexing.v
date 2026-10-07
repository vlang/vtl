module autograd

import vtl

// take_along_axis gathers values and tracks the inverse scatter in backprop.
// Repeated indices accumulate their gradients; non-axis broadcast dimensions
// are reduced into the corresponding source coordinates.
pub fn (v &Variable[T]) take_along_axis(indices &vtl.Tensor[int], axis int) !&Variable[T] {
	value := v.value.take_along_axis[T](indices, axis)!
	mut result := variable[T](v.context, value, requires_grad: v.requires_grad)
	if v.requires_grad {
		axis_index := if axis < 0 { axis + v.value.rank() } else { axis }
		saved_indices := indices.copy(.row_major)
		gate := &TakeAlongAxisGate[T]{
			indices:     saved_indices
			input_shape: v.value.shape.clone()
			axis:        axis_index
		}
		result.grad = vtl.zeros_like[T](value)
		register[T]('TakeAlongAxis', voidptr(gate), take_along_axis_backward_dispatch[T], result,
			[v])!
	}
	return result
}

struct TakeAlongAxisGate[T] {
	indices     &vtl.Tensor[int]
	input_shape []int
	axis        int
}

fn (g &TakeAlongAxisGate[T]) backward(gradient &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	mut input_gradient := vtl.zeros[T](g.input_shape)
	mut input_index := []int{len: gradient.rank()}
	mut index_index := []int{len: gradient.rank()}
	for flat_index in 0 .. gradient.size {
		output_index := gradient.nth_index(flat_index)
		for dimension in 0 .. gradient.rank() {
			input_index[dimension] = if dimension != g.axis && g.input_shape[dimension] == 1 {
				0
			} else {
				output_index[dimension]
			}
			index_index[dimension] = if g.indices.shape[dimension] == 1 {
				0
			} else {
				output_index[dimension]
			}
		}
		selected := g.indices.get(index_index)
		input_index[g.axis] = if selected < 0 {
			selected + g.input_shape[g.axis]
		} else {
			selected
		}
		input_gradient.set(input_index, input_gradient.get(input_index) + gradient.get(output_index))
	}
	return [input_gradient]
}

fn take_along_axis_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_gate := unsafe { &TakeAlongAxisGate[T](gate) }
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := typed_gate.backward(typed_payload.variable.grad)!
	return tensor_ptrs_to_voidptrs[T](tensors)
}
