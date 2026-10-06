module vtl

// UniqueAxisResult contains unique axis slices and their source metadata.
pub struct UniqueAxisResult[T] {
pub:
	values        &Tensor[T]
	counts        &Tensor[int]
	first_indices &Tensor[int]
	inverse       &Tensor[int]
}

// unique_axis returns lexicographically sorted unique slices along axis.
pub fn unique_axis[T](t &Tensor[T], axis int) !&Tensor[T] {
	result := unique_axis_result[T](t, axis)!
	return result.values
}

// unique_axis_result returns unique slices, occurrence counts, first source
// axis positions, and inverse indices aligned to the input axis.
pub fn unique_axis_result[T](t &Tensor[T], axis int) !UniqueAxisResult[T] {
	rank := t.rank()
	if rank == 0 {
		return error('unique_axis requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('unique_axis axis ${axis} is out of bounds for rank ${rank}')
	}
	axis_size := t.shape[axis_index]
	mut output_shape := t.shape.clone()
	if axis_size == 0 {
		mut values := empty[T](output_shape, memory: .row_major)
		return UniqueAxisResult[T]{
			values:        values
			counts:        from_1d[int]([]int{})!
			first_indices: from_1d[int]([]int{})!
			inverse:       from_1d[int]([]int{})!
		}
	}
	slice_size := t.size / axis_size
	mut slices := [][]T{len: axis_size}
	for axis_value in 0 .. axis_size {
		slices[axis_value] = []T{len: slice_size}
	}
	for flat_index in 0 .. t.size {
		index := t.nth_index(flat_index)
		axis_value := index[axis_index]
		slice_index := unique_axis_slice_index(index, t.shape, axis_index)
		slices[axis_value][slice_index] = t.get(index)
	}
	mut order := []int{len: axis_size}
	for i in 0 .. axis_size {
		order[i] = i
	}
	order.sort_with_compare(fn [slices] [T](a &int, b &int) int {
		return compare_unique_axis_slices[T](slices[*a], slices[*b])
	})
	mut unique_positions := []int{cap: axis_size}
	mut counts := []int{cap: axis_size}
	mut first_indices := []int{cap: axis_size}
	mut inverse := []int{len: axis_size}
	for axis_value in order {
		if unique_positions.len == 0
			|| compare_unique_axis_slices[T](slices[unique_positions[unique_positions.len - 1]], slices[axis_value]) != 0 {
			unique_positions << axis_value
			counts << 1
			first_indices << axis_value
		} else {
			counts[counts.len - 1]++
			if axis_value < first_indices[first_indices.len - 1] {
				first_indices[first_indices.len - 1] = axis_value
			}
		}
		inverse[axis_value] = unique_positions.len - 1
	}
	output_shape[axis_index] = unique_positions.len
	mut values := empty[T](output_shape, memory: .row_major)
	for flat_index in 0 .. values.size {
		output_index := values.nth_index(flat_index)
		unique_index := output_index[axis_index]
		slice_index := unique_axis_slice_index(output_index, output_shape, axis_index)
		values.set(output_index, slices[unique_positions[unique_index]][slice_index])
	}
	return UniqueAxisResult[T]{
		values:        values
		counts:        from_1d[int](counts)!
		first_indices: from_1d[int](first_indices)!
		inverse:       from_1d[int](inverse)!
	}
}

fn unique_axis_slice_index(index []int, shape []int, axis int) int {
	mut flat_index := 0
	for dimension in 0 .. shape.len {
		if dimension != axis {
			flat_index = flat_index * shape[dimension] + index[dimension]
		}
	}
	return flat_index
}

fn compare_unique_axis_slices[T](a []T, b []T) int {
	for i in 0 .. a.len {
		comparison := compare_sort_values[T](a[i], b[i])
		if comparison != 0 {
			return comparison
		}
	}
	return 0
}
