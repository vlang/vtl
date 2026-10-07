module vtl

// count_nonzero counts non-zero values across every tensor dimension.
pub fn count_nonzero[T](t &Tensor[T]) int {
	mut count := 0
	for flat_index in 0 .. t.size {
		if td[T](t.get_nth[T](flat_index)).bool() {
			count++
		}
	}
	return count
}

// count_nonzero_axis counts non-zero values along one axis. By default the
// reduced axis is removed; set keepdims to retain it with length one.
pub fn count_nonzero_axis[T](t &Tensor[T], axis int, keepdims bool) !&Tensor[int] {
	rank := t.rank()
	if rank == 0 {
		return error('count_nonzero_axis: axis requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('count_nonzero_axis: axis ${axis} out of bounds for rank ${rank}')
	}
	mut output_shape := []int{cap: if keepdims { rank } else { rank - 1 }}
	for dim, dimension in t.shape {
		if dim == axis_index {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << dimension
		}
	}
	mut counts := []int{len: size_from_shape(output_shape)}
	for flat_index in 0 .. t.size {
		if !td[T](t.get_nth[T](flat_index)).bool() {
			continue
		}
		coordinates := t.nth_index(flat_index)
		mut output_flat_index := 0
		mut output_stride := 1
		mut output_dim := output_shape.len - 1
		for dim := rank - 1; dim >= 0; dim-- {
			if dim == axis_index {
				if keepdims {
					output_dim--
				}
				continue
			}
			output_flat_index += coordinates[dim] * output_stride
			output_stride *= output_shape[output_dim]
			output_dim--
		}
		counts[output_flat_index]++
	}
	return from_array[int](counts, output_shape)
}

// argwhere returns the coordinates of non-zero elements as a row-major tensor
// with shape [number of matches, input rank].
pub fn argwhere[T](t &Tensor[T]) !&Tensor[int] {
	mut coordinates := []int{}
	mut matches := 0
	mut index := []int{len: t.rank()}
	for flat_index in 0 .. t.size {
		if !td[T](t.get_nth(flat_index)).bool() {
			continue
		}
		decode_flat_coordinate(flat_index, t.shape, mut index)
		coordinates << index
		matches++
	}
	return from_array[int](coordinates, [matches, t.rank()])
}

// nonzero returns one index tensor per input axis, matching NumPy's tuple of
// coordinate arrays. A nonzero scalar is treated as a one-dimensional value
// with index zero, following NumPy's scalar promotion behavior.
pub fn nonzero[T](t &Tensor[T]) ![]&Tensor[int] {
	if t.rank() == 0 {
		if !td[T](t.get_nth(0)).bool() {
			return [from_1d[int]([]int{})!]
		}
		return [from_1d[int]([0])!]
	}
	mut axis_values := [][]int{len: t.rank()}
	mut coordinate := []int{len: t.rank()}
	for flat_index in 0 .. t.size {
		if !td[T](t.get_nth(flat_index)).bool() {
			continue
		}
		decode_flat_coordinate(flat_index, t.shape, mut coordinate)
		for axis, value in coordinate {
			axis_values[axis] << value
		}
	}
	mut indices := []&Tensor[int]{cap: t.rank()}
	for axis in 0 .. t.rank() {
		indices << from_1d[int](axis_values[axis])!
	}
	return indices
}

fn decode_flat_coordinate(flat_index int, shape []int, mut coordinate []int) {
	mut remaining := flat_index
	for dim := shape.len - 1; dim >= 0; dim-- {
		coordinate[dim] = remaining % shape[dim]
		remaining /= shape[dim]
	}
}

// advanced_index selects values using one integer coordinate tensor per input
// axis. Coordinate tensors are broadcast together, and each output element is
// read from the matching coordinate tuple. The result is an independent
// row-major copy. This API handles full coordinate tuples; mixing coordinate
// tensors with slices or scalar indices is not supported here.
pub fn advanced_index[T](t &Tensor[T], indices []&Tensor[int]) !&Tensor[T] {
	if t.rank() == 0 {
		if indices.len != 0 {
			return error('advanced_index: scalar tensors do not accept axis indices')
		}
		mut result := empty[T]([], memory: .row_major)
		result.data.data[0] = t.get_nth[T](0)
		return result
	}
	if indices.len != t.rank() {
		return error('advanced_index: expected ${t.rank()} coordinate tensors, got ${indices.len}')
	}
	broadcasted := broadcast_n[int](indices)!
	output_shape := broadcasted[0].shape
	mut result := empty[T](output_shape, memory: .row_major)
	mut output_index := []int{len: output_shape.len}
	mut input_index := []int{len: t.rank()}
	for flat_index in 0 .. result.size {
		decode_flat_coordinate(flat_index, output_shape, mut output_index)
		for axis, index_tensor in broadcasted {
			selected := index_tensor.get(output_index)
			normalized := if selected < 0 { selected + t.shape[axis] } else { selected }
			if normalized < 0 || normalized >= t.shape[axis] {
				return error('advanced_index: index ${selected} is out of range for axis ${axis} with size ${t.shape[axis]}')
			}
			input_index[axis] = normalized
		}
		result.data.data[flat_index] = t.get(input_index)
	}
	return result
}

// UniqueCounts contains sorted unique values and their occurrence counts.
pub struct UniqueCounts[T] {
pub:
	values &Tensor[T]
	counts &Tensor[int]
}

// unique returns the sorted unique values in a flattened copy of the tensor.
// Floating-point NaNs are treated as one value and sort after other values.
pub fn unique[T](t &Tensor[T]) !&Tensor[T] {
	values, _ := sorted_unique_values_counts[T](t)!
	return from_1d[T](values)
}

// unique_counts returns sorted unique values and the number of occurrences of
// each value in the flattened input tensor.
pub fn unique_counts[T](t &Tensor[T]) !UniqueCounts[T] {
	values, counts := sorted_unique_values_counts[T](t)!
	return UniqueCounts[T]{
		values: from_1d[T](values)!
		counts: from_1d[int](counts)!
	}
}

// unique_inverse returns the unique-value index for each flattened input
// element. Apply it to `unique(t)` to reconstruct the input values.
pub fn unique_inverse[T](t &Tensor[T]) !&Tensor[int] {
	values, _ := sorted_unique_values_counts[T](t)!
	mut inverse := []int{len: t.size}
	for input_index in 0 .. t.size {
		value := t.get_nth[T](input_index)
		inverse[input_index] = find_unique_value_index[T](values, value)
	}
	return from_1d[int](inverse)
}

// unique_first_indices returns the first flattened input index for each
// sorted unique value.
pub fn unique_first_indices[T](t &Tensor[T]) !&Tensor[int] {
	values, _ := sorted_unique_values_counts[T](t)!
	mut first_indices := []int{len: values.len, init: -1}
	for input_index in 0 .. t.size {
		value := t.get_nth[T](input_index)
		unique_index := find_unique_value_index[T](values, value)
		if first_indices[unique_index] < 0 {
			first_indices[unique_index] = input_index
		}
	}
	return from_1d[int](first_indices)
}

fn sorted_unique_values_counts[T](t &Tensor[T]) !([]T, []int) {
	mut sorted := []T{len: t.size}
	for i in 0 .. t.size {
		sorted[i] = t.get_nth[T](i)
	}
	sorted.sort_with_compare(fn [T](a &T, b &T) int {
		return compare_sort_values[T](*a, *b)
	})
	mut values := []T{cap: sorted.len}
	mut counts := []int{cap: sorted.len}
	for value in sorted {
		if values.len == 0 || compare_sort_values[T](values[values.len - 1], value) != 0 {
			values << value
			counts << 1
		} else {
			counts[counts.len - 1]++
		}
	}
	return values, counts
}

fn find_unique_value_index[T](values []T, target T) int {
	mut low := 0
	mut high := values.len
	for low < high {
		mid := low + (high - low) / 2
		if compare_sort_values[T](values[mid], target) < 0 {
			low = mid + 1
		} else {
			high = mid
		}
	}
	return low
}

// take returns a copy of the tensor with the values at `indices` selected
// along `axis`. Negative axes and negative indices count from the end.
// The indices are a one-dimensional list; use slice for range-based views.
pub fn (t &Tensor[T]) take[T](indices []int, axis int) !&Tensor[T] {
	rank := t.rank()
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('take axis ${axis} is out of range for rank ${rank}')
	}
	mut normalized_indices := []int{cap: indices.len}
	for index in indices {
		normalized := if index < 0 { index + t.shape[axis_index] } else { index }
		if normalized < 0 || normalized >= t.shape[axis_index] {
			return error('take index ${index} is out of range for axis size ${t.shape[axis_index]}')
		}
		normalized_indices << normalized
	}

	mut output_shape := t.shape.clone()
	output_shape[axis_index] = indices.len
	mut result := empty[T](output_shape, memory: t.memory)
	for flat_index in 0 .. result.size {
		mut output_index := result.nth_index(flat_index)
		mut input_index := output_index.clone()
		input_index[axis_index] = normalized_indices[output_index[axis_index]]
		result.set(output_index, t.get(input_index))
	}
	return result
}

// take_nd gathers values along one axis using an index tensor. The index
// tensor's dimensions replace the selected axis in the output shape.
pub fn (t &Tensor[T]) take_nd[T](indices &Tensor[int], axis int) !&Tensor[T] {
	rank := t.rank()
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('take_nd axis ${axis} is out of range for rank ${rank}')
	}
	mut output_shape := []int{cap: rank - 1 + indices.rank()}
	output_shape << t.shape[..axis_index]
	output_shape << indices.shape
	output_shape << t.shape[axis_index + 1..]
	mut result := empty[T](output_shape, memory: t.memory)
	mut input_index := []int{len: rank}
	mut indices_index := []int{len: indices.rank()}
	mut output_index := []int{len: output_shape.len}
	for flat_index in 0 .. result.size {
		mut remaining := flat_index
		for output_axis := output_shape.len - 1; output_axis >= 0; output_axis-- {
			output_index[output_axis] = remaining % output_shape[output_axis]
			remaining /= output_shape[output_axis]
		}
		for input_axis in 0 .. axis_index {
			input_index[input_axis] = output_index[input_axis]
		}
		for index_axis in 0 .. indices.rank() {
			indices_index[index_axis] = output_index[axis_index + index_axis]
		}
		selected := indices.get(indices_index)
		normalized := if selected < 0 { selected + t.shape[axis_index] } else { selected }
		if normalized < 0 || normalized >= t.shape[axis_index] {
			return error('take_nd index ${selected} is out of range for axis size ${t.shape[axis_index]}')
		}
		input_index[axis_index] = normalized
		for input_axis in axis_index + 1 .. rank {
			input_index[input_axis] = output_index[input_axis - 1 + indices.rank()]
		}
		result.data.data[result.offset_index(output_index)] = t.get(input_index)
	}
	return result
}

// take_flat gathers from the row-major logical flattening of a tensor and
// preserves the shape of the index tensor.
pub fn (t &Tensor[T]) take_flat[T](indices &Tensor[int]) !&Tensor[T] {
	flattened := t.ravel[T]()!
	mut result := empty[T](indices.shape, memory: .row_major)
	for flat_index in 0 .. indices.size {
		selected := indices.get_nth[int](flat_index)
		normalized := if selected < 0 { selected + flattened.size } else { selected }
		if normalized < 0 || normalized >= flattened.size {
			return error('take_flat index ${selected} is out of range for flattened size ${flattened.size}')
		}
		result.data.data[flat_index] = flattened.get_nth(normalized)
	}
	return result
}

// take_along_axis gathers values using an index tensor with the same rank as
// the receiver. Dimensions must match except along `axis`.
pub fn (t &Tensor[T]) take_along_axis[T](indices &Tensor[int], axis int) !&Tensor[T] {
	rank := t.rank()
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('take_along_axis axis ${axis} is out of range for rank ${rank}')
	}
	if indices.rank() != rank {
		return error('take_along_axis index tensor must have rank ${rank}')
	}
	for dimension in 0 .. rank {
		if dimension != axis_index && indices.shape[dimension] != t.shape[dimension] {
			return error('take_along_axis index shape must match tensor shape outside the selected axis')
		}
	}

	mut result := empty[T](indices.shape, memory: t.memory)
	for flat_index in 0 .. result.size {
		result_index := result.nth_index(flat_index)
		mut input_index := result_index.clone()
		selected := indices.get(result_index)
		selected_index := if selected < 0 { selected + t.shape[axis_index] } else { selected }
		if selected_index < 0 || selected_index >= t.shape[axis_index] {
			return error('take_along_axis index ${selected} is out of range for axis size ${t.shape[axis_index]}')
		}
		input_index[axis_index] = selected_index
		result.set(result_index, t.get(input_index))
	}
	return result
}

// get returns a scalar value from a Tensor at the provided index

// get exposes this operation as part of the public API.

// get exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) get[T](index []int) T {
	offset := t.offset_index(index)
	return t.data.get[T](offset)
}

// get_nth returns a scalar value from a Tensor at the provided index

// get_nth exposes this operation as part of the public API.

// get_nth exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) get_nth[T](n int) T {
	if t.is_row_major_contiguous() && n >= 0 && n < t.size {
		return t.data.get[T](n)
	}
	index := t.nth_index(n)
	return t.get[T](index)
}

// offset_index returns the index to a Tensor's data at
// a given index

// offset_index exposes this operation as part of the public API.

// offset_index exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) offset_index[T](index []int) int {
	mut offset := 0
	for i in 0 .. t.rank() {
		mut j := index[i]
		if j < 0 {
			j += t.shape[i]
		}
		offset += j * t.strides[i]
		if t.strides[i] < 0 {
			offset += t.shape[i] - 1
		}
	}

	if offset < 0 {
		offset = t.size - 1 + offset
	}

	return offset
}

// nth_index returns the nth index of a Tensor's shape
// for `n == 2` and a `shape` of `[2, 2]` the _nth index_ is `[1, 0]`
// and for a `shape` of `[2, 3]` and `n == 3` the _nth index_ is `[0, 1, 1]`
// in sorted order.
pub fn (t &Tensor[T]) nth_index[T](n int) []int {
	rank := t.rank()
	mut index := []int{len: rank}
	for i in 0 .. rank {
		index[i] = 0
	}
	mut i := 0
	for {
		if i == n {
			return index
		}
		i += 1
		for j := rank - 1; j >= 0; j -= 1 {
			if index[j] < t.shape[j] - 1 {
				index[j] += 1
				break
			}
			index[j] = 0
		}
	}

	return index
}
