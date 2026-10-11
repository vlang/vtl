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
		minimum := t.min_axes[T](axes, keepdims)!
		maximum := t.max_axes[T](axes, keepdims)!
		return maximum - minimum
	} $else {
		return error('ptp requires an integer or floating-point tensor')
	}
}
