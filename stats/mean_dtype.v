module stats

import math
import vtl

// mean_as computes the mean using U for accumulation and the returned value.
// Empty floating-point inputs return NaN; empty integer inputs return zero.
pub fn mean_as[T, U](t &vtl.Tensor[T]) U {
	if t.size == 0 {
		return empty_mean_value[U]()
	}
	return sum_as[T, U](t) / vtl.cast[U](t.size)
}

// mean_along_axis_as computes the mean along one axis using U for accumulation
// and output. The reduced axis is retained when keepdims is true.
pub fn mean_along_axis_as[T, U](t &vtl.Tensor[T], axis int, keepdims bool) !&vtl.Tensor[U] {
	sums := sum_along_axis_as[T, U](t, axis, keepdims)!
	rank := t.rank()
	axis_index := if axis < 0 { axis + rank } else { axis }
	return divide_mean_sums[U](sums, t.shape[axis_index])
}

// mean_along_axes_as computes the mean over several axes using U for
// accumulation and output. An empty axes list converts each value to U.
pub fn mean_along_axes_as[T, U](t &vtl.Tensor[T], axes []int, keepdims bool) !&vtl.Tensor[U] {
	sums := sum_along_axes_as[T, U](t, axes, keepdims)!
	mut count := 1
	for axis in axes {
		axis_index := if axis < 0 { axis + t.rank() } else { axis }
		count *= t.shape[axis_index]
	}
	return divide_mean_sums[U](sums, count)
}

fn divide_mean_sums[U](sums &vtl.Tensor[U], count int) !&vtl.Tensor[U] {
	mut result := vtl.empty[U](sums.shape, memory: .row_major)
	if count == 0 {
		value := empty_mean_value[U]()
		for index in 0 .. result.size {
			result.set_nth(index, value)
		}
		return result
	}
	divisor := vtl.cast[U](count)
	for index in 0 .. result.size {
		result.set_nth(index, sums.get_nth(index) / divisor)
	}
	return result
}

fn empty_mean_value[U]() U {
	$if U is f32 || U is f64 {
		return U(math.nan())
	} $else {
		return vtl.cast[U](0)
	}
}
