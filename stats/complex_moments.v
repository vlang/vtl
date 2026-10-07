module stats

import math
import math.complex
import vtl

// complex_mean_along_axis computes the arithmetic mean along one axis.
// The output remains complex128 and follows the requested keepdims shape.
pub fn complex_mean_along_axis(t &vtl.Tensor[complex.Complex], axis int, keepdims bool) !&vtl.Tensor[complex.Complex] {
	return complex_moments(t, [axis], keepdims)
}

// complex_variance_along_axis computes the real variance E(|z-mean(z)|^2).
// ddof is subtracted from the observation count; insufficient samples produce NaN.
pub fn complex_variance_along_axis(t &vtl.Tensor[complex.Complex], axis int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	if ddof < 0 {
		return error('complex_variance_along_axis: ddof must be non-negative')
	}
	return complex_moment_variance(t, [axis], ddof, keepdims)
}

// complex_std_along_axis computes the square root of complex variance.
pub fn complex_std_along_axis(t &vtl.Tensor[complex.Complex], axis int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	variances := complex_variance_along_axis(t, axis, ddof, keepdims)!
	return variances.map[f64](fn (value f64, _ []int) f64 { return math.sqrt(value) })
}

// complex_mean_along_axes computes the arithmetic mean over several axes.
pub fn complex_mean_along_axes(t &vtl.Tensor[complex.Complex], axes []int, keepdims bool) !&vtl.Tensor[complex.Complex] {
	return complex_moments(t, axes, keepdims)
}

// complex_variance_along_axes computes real E(|z-mean(z)|^2) over several axes.
pub fn complex_variance_along_axes(t &vtl.Tensor[complex.Complex], axes []int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	if ddof < 0 {
		return error('complex_variance_along_axes: ddof must be non-negative')
	}
	return complex_moment_variance(t, axes, ddof, keepdims)
}

// complex_std_along_axes computes the square root of complex variance over axes.
pub fn complex_std_along_axes(t &vtl.Tensor[complex.Complex], axes []int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	variances := complex_variance_along_axes(t, axes, ddof, keepdims)!
	return variances.map[f64](fn (value f64, _ []int) f64 { return math.sqrt(value) })
}

fn complex_moments(t &vtl.Tensor[complex.Complex], axes []int, keepdims bool) !&vtl.Tensor[complex.Complex] {
	shape, reduced := complex_reduction_shape(t.shape, axes, keepdims)!
	mut sums := []complex.Complex{len: complex_output_size(shape)}
	mut counts := []int{len: sums.len}
	for flat in 0 .. t.size {
		out := complex_output_index(flat, t.shape, reduced)
		value := t.get_nth(flat)
		sums[out] = complex.Complex{ re: sums[out].re + value.re, im: sums[out].im + value.im }
		counts[out]++
	}
	for i in 0 .. sums.len {
		if counts[i] == 0 {
			sums[i] = complex.Complex{ re: math.nan(), im: math.nan() }
		} else {
			sums[i] = complex.Complex{ re: sums[i].re / f64(counts[i]), im: sums[i].im / f64(counts[i]) }
		}
	}
	return vtl.from_array[complex.Complex](sums, shape)
}

fn complex_moment_variance(t &vtl.Tensor[complex.Complex], axes []int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	shape, reduced := complex_reduction_shape(t.shape, axes, keepdims)!
	output_size := complex_output_size(shape)
	mut means := []complex.Complex{len: output_size}
	mut counts := []int{len: output_size}
	for flat in 0 .. t.size {
		out := complex_output_index(flat, t.shape, reduced)
		value := t.get_nth(flat)
		means[out] = complex.Complex{ re: means[out].re + value.re, im: means[out].im + value.im }
		counts[out]++
	}
	for i in 0 .. output_size {
		if counts[i] > 0 {
			means[i] = complex.Complex{ re: means[i].re / f64(counts[i]), im: means[i].im / f64(counts[i]) }
		}
	}
	mut squared_deviations := []f64{len: output_size}
	for flat in 0 .. t.size {
		out := complex_output_index(flat, t.shape, reduced)
		value := t.get_nth(flat)
		dr := value.re - means[out].re
		di := value.im - means[out].im
		squared_deviations[out] += dr * dr + di * di
	}
	for i in 0 .. output_size {
		denominator := counts[i] - ddof
		squared_deviations[i] = if denominator <= 0 {
			math.nan()
		} else {
			squared_deviations[i] / f64(denominator)
		}
	}
	return vtl.from_array[f64](squared_deviations, shape)
}

fn complex_reduction_shape(input_shape []int, axes []int, keepdims bool) !([]int, []bool) {
	rank := input_shape.len
	if rank == 0 && axes.len > 0 {
		return error('complex moment reduction requires a tensor with at least one dimension')
	}
	mut reduced := []bool{len: rank}
	for axis in axes {
		index := if axis < 0 { axis + rank } else { axis }
		if index < 0 || index >= rank {
			return error('axis ${axis} out of bounds for rank ${rank}')
		}
		if reduced[index] {
			return error('axis ${axis} appears more than once')
		}
		reduced[index] = true
	}
	mut shape := []int{cap: rank}
	for i, size in input_shape {
		if reduced[i] {
			if keepdims { shape << 1 }
		} else {
			shape << size
		}
	}
	return shape, reduced
}

fn complex_output_size(shape []int) int {
	mut size := 1
	for dimension in shape { size *= dimension }
	return size
}

fn complex_output_index(flat int, input_shape []int, reduced []bool) int {
	mut remainder := flat
	mut output := 0
	mut output_stride := 1
	for dimension := input_shape.len - 1; dimension >= 0; dimension-- {
		coordinate := remainder % input_shape[dimension]
		remainder /= input_shape[dimension]
		if !reduced[dimension] {
			output += coordinate * output_stride
			output_stride *= input_shape[dimension]
		}
	}
	return output
}
