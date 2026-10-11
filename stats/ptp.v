module stats

import vtl
import math
import simd

// ptp returns the range (maximum minus minimum) of all tensor values.
// Empty inputs and unordered element types return an error.
@[direct_array_access]
pub fn ptp[T](t &vtl.Tensor[T]) !T {
	if t.size == 0 {
		return error('ptp cannot reduce an empty tensor')
	}
	$if T is $int || T is $float {
		if t.is_row_major_contiguous() && t.data.data.len == t.size {
			$if T is f32 {
				return ptp_contiguous_f32(t.data.data)
			} $else $if T is f64 {
				return ptp_contiguous_f64(t.data.data)
			}
		}
		mut minimum := t.get_nth[T](0)
		mut maximum := minimum
		$if T is $float {
			if math.is_nan(f64(minimum)) {
				return minimum - minimum
			}
		}
		for i in 1 .. t.size {
			value := if t.is_row_major_contiguous() && t.data.data.len == t.size {
				t.data.data[i]
			} else {
				t.get_nth[T](i)
			}
			$if T is $float {
				if math.is_nan(f64(value)) {
					return value - value
				}
			}
			if value < minimum {
				minimum = value
			}
			if value > maximum {
				maximum = value
			}
		}
		return maximum - minimum
	} $else {
		return error('ptp requires an integer or floating-point tensor')
	}
}

@[direct_array_access]
fn ptp_contiguous_f32(values []f32) f32 {
	mut minimum := values[0]
	mut maximum := minimum
	if math.is_nan(minimum) {
		return minimum - minimum
	}
	mut offset := 0
	mut min0 := simd.splat_f32x8(minimum)
	mut min1 := min0
	mut max0 := min0
	mut max1 := min0
	for ; offset + 16 <= values.len; offset += 16 {
		block0 := simd.load_f32x8_at(values, offset)
		block1 := simd.load_f32x8_at(values, offset + 8)
		if block0.ne(block0).any() || block1.ne(block1).any() {
			return f32(math.nan())
		}
		min0 = min0.minimum(block0)
		min1 = min1.minimum(block1)
		max0 = max0.maximum(block0)
		max1 = max1.maximum(block1)
	}
	minimum = min0.minimum(min1).min()
	maximum = max0.maximum(max1).max()
	for i in offset .. values.len {
		value := values[i]
		if math.is_nan(value) {
			return f32(math.nan())
		}
		if value < minimum {
			minimum = value
		}
		if value > maximum {
			maximum = value
		}
	}
	return maximum - minimum
}

@[direct_array_access]
fn ptp_contiguous_f64(values []f64) f64 {
	mut minimum := values[0]
	mut maximum := minimum
	if math.is_nan(minimum) {
		return minimum - minimum
	}
	mut offset := 0
	mut min0 := simd.splat_f64x4(minimum)
	mut min1 := min0
	mut max0 := min0
	mut max1 := min0
	for ; offset + 8 <= values.len; offset += 8 {
		block0 := simd.load_f64x4_at(values, offset)
		block1 := simd.load_f64x4_at(values, offset + 4)
		if block0.ne(block0).any() || block1.ne(block1).any() {
			return math.nan()
		}
		min0 = min0.minimum(block0)
		min1 = min1.minimum(block1)
		max0 = max0.maximum(block0)
		max1 = max1.maximum(block1)
	}
	minimum = min0.minimum(min1).min()
	maximum = max0.maximum(max1).max()
	for i in offset .. values.len {
		value := values[i]
		if math.is_nan(value) {
			return math.nan()
		}
		if value < minimum {
			minimum = value
		}
		if value > maximum {
			maximum = value
		}
	}
	return maximum - minimum
}

@[direct_array_access]
fn ptp_contiguous_minmax_f32(values []f32) (f32, f32, bool) {
	nan := f32(math.nan())
	if math.is_nan(values[0]) {
		return nan, nan, true
	}
	mut offset := 0
	mut min0 := simd.splat_f32x8(values[0])
	mut min1 := min0
	mut max0 := min0
	mut max1 := min0
	for ; offset + 16 <= values.len; offset += 16 {
		block0 := simd.load_f32x8_at(values, offset)
		block1 := simd.load_f32x8_at(values, offset + 8)
		if block0.ne(block0).any() || block1.ne(block1).any() {
			return nan, nan, true
		}
		min0 = min0.minimum(block0)
		min1 = min1.minimum(block1)
		max0 = max0.maximum(block0)
		max1 = max1.maximum(block1)
	}
	minimum := min0.minimum(min1).min()
	maximum := max0.maximum(max1).max()
	mut min_value := minimum
	mut max_value := maximum
	for i in offset .. values.len {
		value := values[i]
		if math.is_nan(value) {
			return nan, nan, true
		}
		if value < min_value {
			min_value = value
		}
		if value > max_value {
			max_value = value
		}
	}
	return min_value, max_value, false
}

@[direct_array_access]
fn ptp_contiguous_minmax_f64(values []f64) (f64, f64, bool) {
	nan := math.nan()
	if math.is_nan(values[0]) {
		return nan, nan, true
	}
	mut offset := 0
	mut min0 := simd.splat_f64x4(values[0])
	mut min1 := min0
	mut max0 := min0
	mut max1 := min0
	for ; offset + 8 <= values.len; offset += 8 {
		block0 := simd.load_f64x4_at(values, offset)
		block1 := simd.load_f64x4_at(values, offset + 4)
		if block0.ne(block0).any() || block1.ne(block1).any() {
			return nan, nan, true
		}
		min0 = min0.minimum(block0)
		min1 = min1.minimum(block1)
		max0 = max0.maximum(block0)
		max1 = max1.maximum(block1)
	}
	minimum := min0.minimum(min1).min()
	maximum := max0.maximum(max1).max()
	mut min_value := minimum
	mut max_value := maximum
	for i in offset .. values.len {
		value := values[i]
		if math.is_nan(value) {
			return nan, nan, true
		}
		if value < min_value {
			min_value = value
		}
		if value > max_value {
			max_value = value
		}
	}
	return min_value, max_value, false
}

@[direct_array_access]
fn ptp_contiguous_minmax[T](values []T) (T, T, bool) {
	$if T is f32 {
		minimum, maximum, contains_nan := ptp_contiguous_minmax_f32(values)
		return T(minimum), T(maximum), contains_nan
	} $else $if T is f64 {
		minimum, maximum, contains_nan := ptp_contiguous_minmax_f64(values)
		return T(minimum), T(maximum), contains_nan
	} $else {
		mut minimum := values[0]
		mut maximum := minimum
		for value in values[1..] {
			if value < minimum {
				minimum = value
			}
			if value > maximum {
				maximum = value
			}
		}
		return minimum, maximum, false
	}
}

fn ptp_nan_value[T]() T {
	$if T is f32 {
		return f32(math.nan())
	} $else $if T is f64 {
		return math.nan()
	} $else {
		return T(0)
	}
}

// ptp_along_axis returns maximum minus minimum along one axis. Set keepdims
// to true to retain the reduced axis with length one. Negative axes are valid.
pub fn ptp_along_axis[T](t &vtl.Tensor[T], axis int, keepdims bool) !&vtl.Tensor[T] {
	return ptp_along_axes[T](t, [axis], keepdims)
}

// ptp_along_axes returns maximum minus minimum over several axes. Negative
// axes are valid, duplicate axes and empty reduced dimensions return errors.
// An empty axes list returns zeros with the same shape as the input.
pub fn ptp_along_axes[T](t &vtl.Tensor[T], axes []int, keepdims bool) !&vtl.Tensor[T] {
	$if T is $int || T is $float {
		return ptp_axes_fused[T](t, axes, keepdims)
	} $else {
		return error('ptp requires an integer or floating-point tensor')
	}
}

@[direct_array_access]
fn ptp_axes_fused[T](t &vtl.Tensor[T], axes []int, keepdims bool) !&vtl.Tensor[T] {
	rank := t.rank()
	mut reduced := []bool{len: rank}
	for axis in axes {
		axis_index := if axis < 0 { axis + rank } else { axis }
		if axis_index < 0 || axis_index >= rank {
			return error('axis ${axis} out of bounds for rank ${rank}')
		}
		if reduced[axis_index] {
			return error('duplicate axis ${axis}')
		}
		if t.shape[axis_index] == 0 {
			return error('cannot reduce an empty axis')
		}
		reduced[axis_index] = true
	}

	mut output_shape := []int{cap: rank}
	for axis, dimension in t.shape {
		if reduced[axis] {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << dimension
		}
	}
	if output_shape.len == 0 {
		output_shape = [1]
	}
	mut result := vtl.zeros[T](output_shape)
	if result.size == 0 || t.size == 0 {
		return result
	}
	contiguous := t.is_row_major_contiguous() && t.data.data.len == t.size
	if contiguous && axes.len > 0 {
		mut suffix_start := rank
		for axis := rank - 1; axis >= 0; axis-- {
			if reduced[axis] {
				suffix_start = axis
			} else {
				break
			}
		}
		if axes.len == rank - suffix_start {
			mut reduction_size := 1
			for axis in suffix_start .. rank {
				reduction_size *= t.shape[axis]
			}
			for output_index in 0 .. result.size {
				start := output_index * reduction_size
				result.data.data[output_index] = ptp_contiguous_slice[T](t.data.data[start..start + reduction_size])
			}
			return result
		}
	}

	mut input_strides := []int{len: rank}
	mut input_stride := 1
	for axis := rank - 1; axis >= 0; axis-- {
		input_strides[axis] = input_stride
		input_stride *= t.shape[axis]
	}
	mut output_strides := []int{len: rank}
	mut output_stride := 1
	for axis := rank - 1; axis >= 0; axis-- {
		if !reduced[axis] {
			output_strides[axis] = output_stride
			output_stride *= t.shape[axis]
		}
	}

	mut maximums := []T{len: result.size}
	mut initialized := []bool{len: result.size}
	if contiguous && rank > 1 && reduced[rank - 1] {
		last_dimension := t.shape[rank - 1]
		row_count := t.size / last_dimension
		for row in 0 .. row_count {
			row_start := row * last_dimension
			mut output_index := 0
			for axis in 0 .. rank - 1 {
				if !reduced[axis] {
					coordinate := (row_start / input_strides[axis]) % t.shape[axis]
					output_index += coordinate * output_strides[axis]
				}
			}
			row_end := row_start + last_dimension
			minimum, maximum, contains_nan := ptp_contiguous_minmax[T](t.data.data[row_start..row_end])
			if contains_nan {
				result.data.data[output_index] = ptp_nan_value[T]()
				maximums[output_index] = ptp_nan_value[T]()
				initialized[output_index] = true
			} else if !initialized[output_index] {
				result.data.data[output_index] = minimum
				maximums[output_index] = maximum
				initialized[output_index] = true
			} else if !ptp_value_is_nan[T](result.data.data[output_index]) {
				if minimum < result.data.data[output_index] {
					result.data.data[output_index] = minimum
				}
				if maximum > maximums[output_index] {
					maximums[output_index] = maximum
				}
			}
		}
		for output_index in 0 .. result.size {
			if !ptp_value_is_nan[T](result.data.data[output_index]) {
				result.data.data[output_index] = maximums[output_index] - result.data.data[output_index]
			}
		}
		return result
	}
	for flat_index in 0 .. t.size {
		mut output_index := 0
		for axis in 0 .. rank {
			if !reduced[axis] {
				coordinate := (flat_index / input_strides[axis]) % t.shape[axis]
				output_index += coordinate * output_strides[axis]
			}
		}
		value := if contiguous { t.data.data[flat_index] } else { t.get_nth[T](flat_index) }
		if ptp_value_is_nan[T](value) {
			result.data.data[output_index] = value
			maximums[output_index] = value
			initialized[output_index] = true
			continue
		}
		if !initialized[output_index] {
			result.data.data[output_index] = value
			maximums[output_index] = value
			initialized[output_index] = true
		} else if !ptp_value_is_nan[T](result.data.data[output_index]) {
			if value < result.data.data[output_index] {
				result.data.data[output_index] = value
			}
			if value > maximums[output_index] {
				maximums[output_index] = value
			}
		}
	}

	for output_index in 0 .. result.size {
		if !ptp_value_is_nan[T](result.data.data[output_index]) {
			result.data.data[output_index] = maximums[output_index] - result.data.data[output_index]
		}
	}
	return result
}

@[direct_array_access]
fn ptp_contiguous_slice[T](values []T) T {
	minimum, maximum, contains_nan := ptp_contiguous_minmax[T](values)
	if contains_nan {
		return ptp_nan_value[T]()
	}
	return maximum - minimum
}

fn ptp_value_is_nan[T](value T) bool {
	$if T is $float {
		return math.is_nan(f64(value))
	} $else {
		return false
	}
}
