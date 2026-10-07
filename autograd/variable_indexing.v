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

// scatter_add adds updates at indexed positions and tracks gradients for both
// the source tensor and update values.
pub fn (v &Variable[T]) scatter_add(indices &vtl.Tensor[int], updates &Variable[T], axis int) !&Variable[T] {
	mut value := v.value.copy(.row_major)
	value.scatter_add[T](indices, updates.value, axis)!
	needs_grad := v.requires_grad || updates.requires_grad
	mut result := variable[T](v.context, value, requires_grad: needs_grad)
	if needs_grad {
		axis_index := if axis < 0 { axis + v.value.rank() } else { axis }
		gate := &ScatterAddGate[T]{
			indices: indices.copy(.row_major)
			axis:    axis_index
		}
		result.grad = vtl.zeros_like[T](value)
		register[T]('ScatterAdd', voidptr(gate), scatter_add_backward_dispatch[T], result, [
			v,
			updates,
		])!
	}
	return result
}

struct ScatterAddGate[T] {
	indices &vtl.Tensor[int]
	axis    int
}

fn (g &ScatterAddGate[T]) backward(gradient &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	return [gradient.copy(.row_major), gradient.take_along_axis[T](g.indices, g.axis)!]
}

fn scatter_add_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_gate := unsafe { &ScatterAddGate[T](gate) }
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := typed_gate.backward(typed_payload.variable.grad)!
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// put_along_axis writes updates into a copy and tracks gradients. When multiple
// updates target the same position, only the last update receives its gradient.
pub fn (v &Variable[T]) put_along_axis(indices &vtl.Tensor[int], updates &Variable[T], axis int) !&Variable[T] {
	mut value := v.value.copy(.row_major)
	value.put_along_axis[T](indices, updates.value, axis)!
	needs_grad := v.requires_grad || updates.requires_grad
	mut result := variable[T](v.context, value, requires_grad: needs_grad)
	if needs_grad {
		axis_index := if axis < 0 { axis + v.value.rank() } else { axis }
		gate := &PutAlongAxisGate[T]{
			indices:     indices.copy(.row_major)
			input_shape: v.value.shape.clone()
			axis:        axis_index
		}
		result.grad = vtl.zeros_like[T](value)
		register[T]('PutAlongAxis', voidptr(gate), put_along_axis_backward_dispatch[T], result,
			[v, updates])!
	}
	return result
}

struct PutAlongAxisGate[T] {
	indices     &vtl.Tensor[int]
	input_shape []int
	axis        int
}

fn (g &PutAlongAxisGate[T]) backward(gradient &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	mut input_gradient := gradient.copy(.row_major)
	mut last_update_by_destination := map[int]int{}
	mut destinations := []int{len: g.indices.size}
	for update_position in 0 .. g.indices.size {
		mut destination_index := g.indices.nth_index(update_position)
		selected := g.indices.get(destination_index)
		destination_index[g.axis] = if selected < 0 {
			selected + g.input_shape[g.axis]
		} else {
			selected
		}
		destination := flat_index_from_coordinate(destination_index, g.input_shape)
		destinations[update_position] = destination
		last_update_by_destination[destination] = update_position
	}
	for destination, _ in last_update_by_destination {
		mut destination_index := []int{len: g.input_shape.len}
		mut remaining := destination
		for dimension := g.input_shape.len - 1; dimension >= 0; dimension-- {
			destination_index[dimension] = remaining % g.input_shape[dimension]
			remaining /= g.input_shape[dimension]
		}
		input_gradient.set(destination_index, vtl.cast[T](0))
	}
	mut update_gradient_values := []T{len: g.indices.size}
	for update_position in 0 .. g.indices.size {
		if last_update_by_destination[destinations[update_position]] != update_position {
			continue
		}
		update_index := g.indices.nth_index(update_position)
		selected := g.indices.get(update_index)
		update_index[g.axis] = if selected < 0 {
			selected + g.input_shape[g.axis]
		} else {
			selected
		}
		update_gradient_values[update_position] = gradient.get(update_index)
	}
	update_gradient := vtl.from_array[T](update_gradient_values, g.indices.shape)!
	return [input_gradient, update_gradient]
}

fn put_along_axis_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_gate := unsafe { &PutAlongAxisGate[T](gate) }
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := typed_gate.backward(typed_payload.variable.grad)!
	return tensor_ptrs_to_voidptrs[T](tensors)
}

fn flat_index_from_coordinate(coordinate []int, shape []int) int {
	mut flat_index := 0
	for dimension, value in coordinate {
		flat_index = flat_index * shape[dimension] + value
	}
	return flat_index
}
