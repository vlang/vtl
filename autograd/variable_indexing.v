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

// put replaces row-major flat positions and tracks gradients for the source
// and updates. Short update tensors repeat in the same order as Tensor.put.
pub fn (v &Variable[T]) put(indices &vtl.Tensor[int], updates &Variable[T]) !&Variable[T] {
	return v.put_with_mode(indices, updates, .raise)
}

// put_with_mode is the differentiable counterpart of Tensor.put_with_mode.
pub fn (v &Variable[T]) put_with_mode(indices &vtl.Tensor[int], updates &Variable[T], mode vtl.PutMode) !&Variable[T] {
	mut value := v.value.copy(.row_major)
	value.put_with_mode[T](indices, updates.value, mode)!
	needs_grad := v.requires_grad || updates.requires_grad
	mut result := variable[T](v.context, value, requires_grad: needs_grad)
	if needs_grad {
		mut destinations := []int{cap: indices.size}
		for position in 0 .. indices.size {
			destinations << normalize_flat_put_index(indices.get_nth[int](position), v.value.size, mode)!
		}
		gate := &PutGate[T]{
			destinations:  destinations
			updates_shape: updates.value.shape.clone()
			update_size:   updates.value.size
		}
		result.grad = vtl.zeros_like[T](value)
		register[T]('Put', voidptr(gate), put_backward_dispatch[T], result, [v, updates])!
	}
	return result
}

struct PutGate[T] {
	destinations  []int
	updates_shape []int
	update_size   int
}

fn (g &PutGate[T]) backward(gradient &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	mut input_gradient := gradient.copy(.row_major)
	mut last_update_by_destination := map[int]int{}
	for update_position, destination in g.destinations {
		last_update_by_destination[destination] = update_position
	}
	for destination, _ in last_update_by_destination {
		input_gradient.set_nth[T](destination, vtl.cast[T](0))
	}
	mut update_gradient_values := []T{len: g.update_size}
	for update_position, destination in g.destinations {
		if last_update_by_destination[destination] == update_position {
			update_position_in_values := update_position % g.update_size
			update_gradient_values[update_position_in_values] += gradient.get_nth[T](destination)
		}
	}
	update_gradient := vtl.from_array[T](update_gradient_values, g.updates_shape)!
	return [input_gradient, update_gradient]
}

fn put_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_gate := unsafe { &PutGate[T](gate) }
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := typed_gate.backward(typed_payload.variable.grad)!
	return tensor_ptrs_to_voidptrs[T](tensors)
}

fn normalize_flat_put_index(selected int, size int, mode vtl.PutMode) !int {
	if size == 0 {
		return error('put cannot index an empty tensor')
	}
	return match mode {
		.raise {
			index := if selected < 0 { selected + size } else { selected }
			if index < 0 || index >= size {
				return error('put index ${selected} is out of range for flattened size ${size}')
			}
			index
		}
		.wrap {
			remainder := selected % size
			if remainder < 0 { remainder + size } else { remainder }
		}
		.clip {
			if selected < 0 {
				0
			} else if selected >= size {
				size - 1
			} else {
				selected
			}
		}
	}
}

fn flat_index_from_coordinate(coordinate []int, shape []int) int {
	mut flat_index := 0
	for dimension, value in coordinate {
		flat_index = flat_index * shape[dimension] + value
	}
	return flat_index
}

struct SliceBackwardMapping {
	input_shape []int
	starts      []int
	steps       []int
	output_axes []int
}

fn variable_slice_result[T](parent &Variable[T], value &vtl.Tensor[T], mapping SliceBackwardMapping) !&Variable[T] {
	mut result := variable[T](parent.context, value, requires_grad: parent.requires_grad)
	if parent.requires_grad {
		gate := &SliceGate[T]{ mapping: mapping }
		result.grad = vtl.zeros_like[T](value)
		register[T]('Slice', voidptr(gate), slice_backward_dispatch[T], result, [parent])!
	}
	return result
}

fn slice_backward_mapping(input_shape []int, selectors [][]int, output_shape []int) !SliceBackwardMapping {
	mut starts := []int{len: input_shape.len}
	mut steps := []int{len: input_shape.len, init: 1}
	mut output_axes := []int{len: input_shape.len, init: -1}
	mut output_axis := 0
	for dimension, size in input_shape {
		selector := if dimension < selectors.len { selectors[dimension] } else { []int{} }
		mut start := 0
		mut step := 1
		mut keep_axis := true
		match selector.len {
			0 {}
			1 {
				start = selector[0]
				if start < 0 {
					start += size
				}
				keep_axis = false
			}
			2 {
				start = selector[0]
				mut stop := selector[1]
				if start < 0 {
					start += size
				}
				if stop < 0 {
					stop += size
				}
				keep_axis = start != stop
			}
			3 {
				start = selector[0]
				mut stop := selector[1]
				step = selector[2]
				if start < 0 {
					start += size
				}
				if stop < 0 {
					stop += size
				}
				abs_step := if step < 0 { -step } else { step }
				offset := stop - start
				slice_size := offset / abs_step + offset % abs_step
				keep_axis = slice_size != 0
			}
			else {}
		}
		starts[dimension] = start
		steps[dimension] = step
		if keep_axis {
			if output_axis >= output_shape.len {
				return error('Variable.slice: output shape does not match the input slice mapping')
			}
			output_axes[dimension] = output_axis
			output_axis++
		}
	}
	if output_axis != output_shape.len {
		return error('Variable.slice: output shape does not match the input slice mapping')
	}
	return SliceBackwardMapping{
		input_shape: input_shape.clone()
		starts:      starts
		steps:       steps
		output_axes: output_axes
	}
}

fn slice_hilo_backward_mapping(input_shape []int, starts_in []int, stops_in []int, output_shape []int) !SliceBackwardMapping {
	mut starts := []int{len: input_shape.len}
	mut steps := []int{len: input_shape.len, init: 1}
	mut output_axes := []int{len: input_shape.len, init: -1}
	mut output_axis := 0
	for dimension, size in input_shape {
		mut start := if dimension < starts_in.len { starts_in[dimension] } else { 0 }
		mut stop := if dimension < stops_in.len { stops_in[dimension] } else { size }
		if start < 0 {
			start += size
		}
		if stop < 0 {
			stop += size
		}
		starts[dimension] = start
		if start != stop {
			if output_axis >= output_shape.len {
				return error('Variable.slice_hilo: output shape does not match the input slice mapping')
			}
			output_axes[dimension] = output_axis
			output_axis++
		}
	}
	if output_axis != output_shape.len {
		return error('Variable.slice_hilo: output shape does not match the input slice mapping')
	}
	return SliceBackwardMapping{
		input_shape: input_shape.clone()
		starts:      starts
		steps:       steps
		output_axes: output_axes
	}
}

struct SliceGate[T] {
	mapping SliceBackwardMapping
}

fn (g &SliceGate[T]) backward(gradient &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	mut input_gradient := vtl.zeros[T](g.mapping.input_shape)
	mut input_index := []int{len: g.mapping.input_shape.len}
	for flat_index in 0 .. gradient.size {
		output_index := gradient.nth_index(flat_index)
		for dimension in 0 .. input_index.len {
			input_index[dimension] = g.mapping.starts[dimension]
			if g.mapping.output_axes[dimension] >= 0 {
				axis := g.mapping.output_axes[dimension]
				input_index[dimension] += output_index[axis] * g.mapping.steps[dimension]
			}
		}
		input_gradient.set(input_index, gradient.get(output_index))
	}
	return [input_gradient]
}

fn slice_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_gate := unsafe { &SliceGate[T](gate) }
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := typed_gate.backward(typed_payload.variable.grad)!
	return tensor_ptrs_to_voidptrs[T](tensors)
}
