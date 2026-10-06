module vtl

// diff computes the n-th discrete difference along an axis. Negative axes
// count from the end, and n=0 returns a row-major copy of the input.
pub fn diff[T](t &Tensor[T], n int, axis int) !&Tensor[T] {
	if n < 0 {
		return error('diff: n must be non-negative')
	}
	if t.rank() == 0 {
		return error('diff: input must have at least one dimension')
	}
	axis_index := if axis < 0 { axis + t.rank() } else { axis }
	if axis_index < 0 || axis_index >= t.rank() {
		return error('diff: axis ${axis} out of bounds for rank ${t.rank()}')
	}
	mut result := t.copy(.row_major)
	for _ in 0 .. n {
		mut output_shape := result.shape.clone()
		output_shape[axis_index] = if result.shape[axis_index] > 0 {
			result.shape[axis_index] - 1
		} else {
			0
		}
		mut output := []T{len: size_from_shape(output_shape)}
		for flat_index in 0 .. output.len {
			mut index := []int{len: output_shape.len}
			mut remainder := flat_index
			for dimension := output_shape.len - 1; dimension >= 0; dimension-- {
				index[dimension] = remainder % output_shape[dimension]
				remainder /= output_shape[dimension]
			}
			first := result.get(index)
			index[axis_index]++
			second := result.get(index)
			output[flat_index] = second - first
		}
		result = from_array[T](output, output_shape)!
	}
	return result
}
