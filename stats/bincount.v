module stats

import vtl

// bincount counts occurrences of each non-negative integer in a vector. The
// output length is max(input)+1 or minlength, whichever is larger.
pub fn bincount[T](input &vtl.Tensor[T], minlength int) !&vtl.Tensor[int] {
	indices := validate_bincount_input[T](input, minlength)!
	mut counts := []int{len: bincount_output_length(indices, minlength)}
	for index in indices {
		counts[index]++
	}
	return vtl.from_1d(counts)
}

// bincount_weighted sums one weight per non-negative integer input value.
// It follows NumPy's weighted bincount shape and minlength rules.
pub fn bincount_weighted[T, W](input &vtl.Tensor[T], weights &vtl.Tensor[W], minlength int) !&vtl.Tensor[f64] {
	indices := validate_bincount_input[T](input, minlength)!
	if weights.rank() != 1 || weights.size != indices.len {
		return error('bincount weights must match the one-dimensional input shape')
	}
	mut counts := []f64{len: bincount_output_length(indices, minlength)}
	for i, index in indices {
		counts[index] += bincount_weight_value[W](weights.get_nth(i))!
	}
	return vtl.from_1d(counts)
}

fn validate_bincount_input[T](input &vtl.Tensor[T], minlength int) ![]int {
	if input.rank() != 1 {
		return error('bincount expects a one-dimensional input')
	}
	if minlength < 0 {
		return error('bincount minlength must be non-negative')
	}
	mut indices := []int{len: input.size}
	for position in 0 .. input.size {
		indices[position] = bincount_index[T](input.get_nth(position))!
	}
	return indices
}

fn bincount_index[T](value T) !int {
	$if T is i8 || T is i16 || T is i32 || T is i64 || T is int {
		if value < 0 {
			return error('bincount values must be non-negative')
		}
		index := int(value)
		if index >= max_int {
			return error('bincount value is too large for an output tensor')
		}
		return index
	} $else $if T is u8 || T is u16 || T is u32 || T is u64 {
		if u64(value) >= u64(max_int) {
			return error('bincount value is too large for an output tensor')
		}
		return int(value)
	} $else {
		return error('bincount input must have an integer dtype')
	}
}

fn bincount_weight_value[W](value W) !f64 {
	$if W is i8 || W is i16 || W is i32 || W is i64 || W is int || W is u8 || W is u16 || W is u32 || W is u64 || W is f32 || W is f64 {
		return f64(value)
	} $else {
		return error('bincount weights must have a numeric dtype')
	}
}

fn bincount_output_length(indices []int, minlength int) int {
	mut length := minlength
	for index in indices {
		if index + 1 > length {
			length = index + 1
		}
	}
	return length
}
