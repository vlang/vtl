module vtl

// TriangularOptions selects the diagonal offset for triangular matrix helpers.
@[params]
pub struct TriangularOptions {
pub:
	k int
}

// tril returns a copy with entries above the k-th diagonal of each trailing
// matrix set to zero. Inputs must have at least two dimensions.
pub fn tril[T](input &Tensor[T], options TriangularOptions) !&Tensor[T] {
	return triangular_copy[T](input, options.k, true)
}

// triu returns a copy with entries below the k-th diagonal of each trailing
// matrix set to zero. Inputs must have at least two dimensions.
pub fn triu[T](input &Tensor[T], options TriangularOptions) !&Tensor[T] {
	return triangular_copy[T](input, options.k, false)
}

fn triangular_copy[T](input &Tensor[T], k int, lower bool) !&Tensor[T] {
	if input.rank() < 2 {
		return error('triangular matrix operation requires an input with rank at least 2')
	}
	mut result := input.copy(.row_major)
	mut index := []int{len: input.rank()}
	for flat_index in 0 .. input.size {
		decode_triangular_index(flat_index, input.shape, mut index)
		diagonal_offset := index[index.len - 1] - index[index.len - 2]
		if (lower && diagonal_offset > k) || (!lower && diagonal_offset < k) {
			result.set(index, cast[T](0))
		}
	}
	return result
}

fn decode_triangular_index(flat_index int, shape []int, mut index []int) {
	mut remainder := flat_index
	for axis := shape.len - 1; axis >= 0; axis-- {
		index[axis] = remainder % shape[axis]
		remainder /= shape[axis]
	}
}
