module vtl

// vmap applies a tensor function independently over one input axis and stacks
// the results along the requested output axis. The mapped axis is removed
// before calling f, so f operates on one sample at a time.
pub fn vmap[T](f fn (&Tensor[T]) !&Tensor[T], input &Tensor[T], in_dim int, out_dim int) !&Tensor[T] {
	rank := input.shape.len
	if rank == 0 {
		return error('vmap requires an input with at least one dimension')
	}
	mut input_axis := in_dim
	if input_axis < 0 {
		input_axis += rank
	}
	if input_axis < 0 || input_axis >= rank {
		return error('vmap in_dim ${in_dim} is out of range for rank ${rank}')
	}
	mut output_axis := -1
	mut expected_result_shape := []int{}
	mut outputs := []&Tensor[T]{cap: input.shape[input_axis]}
	mut expected_shape := []int{}
	for mapped_index in 0 .. input.shape[input_axis] {
		mut indices := [][]int{len: rank}
		for axis in 0 .. rank {
			indices[axis] = if axis == input_axis { [mapped_index] } else { []int{} }
		}
		sample := input.slice(...indices)!
		result := f(sample)!
		if mapped_index == 0 {
			output_rank := result.rank() + 1
			output_axis = out_dim
			if output_axis < 0 {
				output_axis += output_rank
			}
			if output_axis < 0 || output_axis >= output_rank {
				return error('vmap out_dim ${out_dim} is out of range for output rank ${output_rank}')
			}
			expected_result_shape = result.shape.clone()
		} else if result.shape != expected_result_shape {
			return error('vmap function returned inconsistent shapes')
		}
		mut expanded_shape := result.shape.clone()
		expanded_shape.insert(output_axis, 1)
		expanded := result.reshape(expanded_shape)!
		if mapped_index == 0 {
			expected_shape = expanded.shape.clone()
		} else if expanded.shape != expected_shape {
			return error('vmap function returned inconsistent shapes')
		}
		outputs << expanded
	}
	if outputs.len == 0 {
		return error('vmap cannot map over an empty axis')
	}
	return concatenate[T](outputs, axis: output_axis)
}
