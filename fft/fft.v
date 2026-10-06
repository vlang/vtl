module fft

import math.complex
import vsl.fft as vsl_fft
import vtl

// rfft computes the one-dimensional real-input discrete Fourier transform.
// It returns the non-negative frequencies as complex f64 values, matching the
// compact layout used by NumPy's rfft. Input tensors may contain f32 or f64.
pub fn rfft[T](input &vtl.Tensor[T]) !&vtl.Tensor[complex.Complex] {
	if input.rank() != 1 {
		return error('rfft expects a one-dimensional tensor')
	}
	if input.size == 0 {
		return error('rfft requires at least one input value')
	}
	$if T is f32 {
		return rfft_f32(unsafe { &vtl.Tensor[f32](input) })
	} $else $if T is f64 {
		return rfft_f64(unsafe { &vtl.Tensor[f64](input) })
	} $else {
		return error('rfft supports f32 and f64 input tensors')
	}
}

// fft computes a one-dimensional discrete Fourier transform of complex f64
// values, returning all positive and negative frequencies in native order.
pub fn fft(input &vtl.Tensor[complex.Complex]) !&vtl.Tensor[complex.Complex] {
	return complex_fft(input, false)
}

// ifft computes the normalized one-dimensional inverse transform of complex
// f64 values.
pub fn ifft(input &vtl.Tensor[complex.Complex]) !&vtl.Tensor[complex.Complex] {
	return complex_fft(input, true)
}

// fftn computes the complex discrete Fourier transform across every axis.
pub fn fftn(input &vtl.Tensor[complex.Complex]) !&vtl.Tensor[complex.Complex] {
	return multidimensional_fft(input, false)
}

// ifftn computes the normalized complex inverse transform across every axis.
pub fn ifftn(input &vtl.Tensor[complex.Complex]) !&vtl.Tensor[complex.Complex] {
	return multidimensional_fft(input, true)
}

// fft2 computes a two-dimensional complex discrete Fourier transform.
pub fn fft2(input &vtl.Tensor[complex.Complex]) !&vtl.Tensor[complex.Complex] {
	if input.rank() != 2 {
		return error('fft2 expects a two-dimensional tensor')
	}
	return multidimensional_fft(input, false)
}

// ifft2 computes the normalized two-dimensional complex inverse transform.
pub fn ifft2(input &vtl.Tensor[complex.Complex]) !&vtl.Tensor[complex.Complex] {
	if input.rank() != 2 {
		return error('ifft2 expects a two-dimensional tensor')
	}
	return multidimensional_fft(input, true)
}

// irfft reconstructs a real signal of `length` samples from its non-negative
// frequency components. The inverse transform is normalized by `length`.
pub fn irfft(input &vtl.Tensor[complex.Complex], length int) !&vtl.Tensor[f64] {
	if input.rank() != 1 {
		return error('irfft expects a one-dimensional frequency tensor')
	}
	if length <= 0 {
		return error('irfft length must be positive')
	}
	expected_bins := length / 2 + 1
	if input.size != expected_bins {
		return error('irfft expects ${expected_bins} frequency bins for length ${length}, got ${input.size}')
	}
	mut packed := []f64{len: length}
	packed[0] = input.get_nth(0).re
	for frequency in 1 .. (length + 1) / 2 {
		value := input.get_nth(frequency)
		packed[2 * frequency - 1] = value.re
		packed[2 * frequency] = value.im
	}
	if length % 2 == 0 && length > 1 {
		packed[length - 1] = input.get_nth(length / 2).re
	}
	plan := vsl_fft.create_plan(packed) or { return error('irfft could not create an FFT plan') }
	defer {
		vsl_fft.destroy_plan(plan)
	}
	if vsl_fft.backward_fft(plan, mut packed) != 0 {
		return error('irfft backend failed to compute the inverse transform')
	}
	for i in 0 .. packed.len {
		packed[i] /= f64(length)
	}
	return vtl.from_1d[f64](packed)
}

fn rfft_f32(input &vtl.Tensor[f32]) !&vtl.Tensor[complex.Complex] {
	mut packed := input.to_array()
	plan := vsl_fft.create_plan(packed) or { return error('rfft could not create an FFT plan') }
	defer {
		vsl_fft.destroy_plan(plan)
	}
	if vsl_fft.forward_fft(plan, mut packed) != 0 {
		return error('rfft backend failed to compute the forward transform')
	}
	return unpack_rfft_f32(packed)
}

fn rfft_f64(input &vtl.Tensor[f64]) !&vtl.Tensor[complex.Complex] {
	mut packed := input.to_array()
	plan := vsl_fft.create_plan(packed) or { return error('rfft could not create an FFT plan') }
	defer {
		vsl_fft.destroy_plan(plan)
	}
	if vsl_fft.forward_fft(plan, mut packed) != 0 {
		return error('rfft backend failed to compute the forward transform')
	}
	return unpack_rfft_f64(packed)
}

fn complex_fft(input &vtl.Tensor[complex.Complex], inverse bool) !&vtl.Tensor[complex.Complex] {
	if input.rank() != 1 {
		return error('fft expects a one-dimensional tensor')
	}
	if input.size == 0 {
		return error('fft requires at least one input value')
	}
	return transform_complex_axis(input, 0, inverse)
}

fn multidimensional_fft(input &vtl.Tensor[complex.Complex], inverse bool) !&vtl.Tensor[complex.Complex] {
	if input.rank() == 0 {
		return error('fftn expects a tensor with at least one dimension')
	}
	if input.size == 0 {
		return error('fftn requires at least one input value')
	}
	mut result := transform_complex_axis(input, 0, inverse)!
	for axis in 1 .. input.rank() {
		result = transform_complex_axis(result, axis, inverse)!
	}
	return result
}

fn transform_complex_axis(input &vtl.Tensor[complex.Complex], axis int, inverse bool) !&vtl.Tensor[complex.Complex] {
	axis_length := input.shape[axis]
	if axis_length <= 0 {
		return error('fft axis length must be positive')
	}
	plan := vsl_fft.create_complex_plan_f64(axis_length)!
	defer {
		vsl_fft.destroy_plan(plan)
	}
	mut output := []complex.Complex{len: input.size}
	mut line := []f64{len: axis_length * 2}
	mut index := []int{len: input.rank()}
	line_count := input.size / axis_length
	for line_index in 0 .. line_count {
		mut remainder := line_index
		for dimension := input.rank() - 1; dimension >= 0; dimension-- {
			if dimension == axis {
				continue
			}
			index[dimension] = remainder % input.shape[dimension]
			remainder /= input.shape[dimension]
		}
		for position in 0 .. axis_length {
			index[axis] = position
			value := input.get(index)
			line[2 * position] = value.re
			line[2 * position + 1] = value.im
		}
		status := if inverse {
			vsl_fft.backward_complex_f64(plan, mut line)
		} else {
			vsl_fft.forward_complex_f64(plan, mut line)
		}
		if status != 0 {
			return error('fft backend failed to compute the transform')
		}
		for position in 0 .. axis_length {
			index[axis] = position
			output_index := row_major_index(index, input.shape)
			factor := if inverse { f64(axis_length) } else { 1.0 }
			output[output_index] = complex.complex(line[2 * position] / factor,
				line[2 * position + 1] / factor)
		}
	}
	return vtl.from_array[complex.Complex](output, input.shape, memory: .row_major)
}

fn row_major_index(index []int, shape []int) int {
	mut flat_index := 0
	for axis in 0 .. shape.len {
		flat_index = flat_index * shape[axis] + index[axis]
	}
	return flat_index
}

fn unpack_rfft_f32(packed []f32) !&vtl.Tensor[complex.Complex] {
	mut frequencies := []complex.Complex{len: packed.len / 2 + 1}
	frequencies[0] = complex.complex(f64(packed[0]), 0)
	for frequency in 1 .. (packed.len + 1) / 2 {
		frequencies[frequency] = complex.complex(f64(packed[2 * frequency - 1]),
			f64(packed[2 * frequency]))
	}
	if packed.len % 2 == 0 && packed.len > 1 {
		frequencies[packed.len / 2] = complex.complex(f64(packed[packed.len - 1]), 0)
	}
	return vtl.from_1d[complex.Complex](frequencies)
}

fn unpack_rfft_f64(packed []f64) !&vtl.Tensor[complex.Complex] {
	mut frequencies := []complex.Complex{len: packed.len / 2 + 1}
	frequencies[0] = complex.complex(packed[0], 0)
	for frequency in 1 .. (packed.len + 1) / 2 {
		frequencies[frequency] = complex.complex(packed[2 * frequency - 1], packed[2 * frequency])
	}
	if packed.len % 2 == 0 && packed.len > 1 {
		frequencies[packed.len / 2] = complex.complex(packed[packed.len - 1], 0)
	}
	return vtl.from_1d[complex.Complex](frequencies)
}
