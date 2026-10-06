module fft

import math
import math.complex
import vtl

// FftNorm selects the forward/inverse scaling convention used by NumPy.
pub enum FftNorm {
	backward
	forward
	ortho
}

// fft_norm computes a one-dimensional complex FFT with an explicit normalization mode.
pub fn fft_norm(input &vtl.Tensor[complex.Complex], norm FftNorm) !&vtl.Tensor[complex.Complex] {
	mut result := fft(input)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, false, input.size))
}

// ifft_norm computes a one-dimensional complex inverse FFT with an explicit normalization mode.
pub fn ifft_norm(input &vtl.Tensor[complex.Complex], norm FftNorm) !&vtl.Tensor[complex.Complex] {
	mut result := ifft(input)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, true, input.size))
}

// fft_axis_norm transforms one complex tensor axis using an explicit normalization mode.
pub fn fft_axis_norm(input &vtl.Tensor[complex.Complex], axis int, norm FftNorm) !&vtl.Tensor[complex.Complex] {
	axis_index := normalize_fft_axis(input.rank(), axis)!
	mut result := fft_axis(input, axis_index)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, false, input.shape[axis_index]))
}

// ifft_axis_norm applies an explicit normalization mode to one complex inverse FFT axis.
pub fn ifft_axis_norm(input &vtl.Tensor[complex.Complex], axis int, norm FftNorm) !&vtl.Tensor[complex.Complex] {
	axis_index := normalize_fft_axis(input.rank(), axis)!
	mut result := ifft_axis(input, axis_index)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, true, input.shape[axis_index]))
}

// fftn_norm transforms every complex tensor axis with an explicit normalization mode.
pub fn fftn_norm(input &vtl.Tensor[complex.Complex], norm FftNorm) !&vtl.Tensor[complex.Complex] {
	mut result := fftn(input)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, false, input.size))
}

// ifftn_norm inversely transforms every complex tensor axis with an explicit normalization mode.
pub fn ifftn_norm(input &vtl.Tensor[complex.Complex], norm FftNorm) !&vtl.Tensor[complex.Complex] {
	mut result := ifftn(input)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, true, input.size))
}

// fft2_norm computes a two-dimensional complex FFT with explicit normalization.
pub fn fft2_norm(input &vtl.Tensor[complex.Complex], norm FftNorm) !&vtl.Tensor[complex.Complex] {
	mut result := fft2(input)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, false, input.size))
}

// ifft2_norm computes a two-dimensional complex inverse FFT with explicit normalization.
pub fn ifft2_norm(input &vtl.Tensor[complex.Complex], norm FftNorm) !&vtl.Tensor[complex.Complex] {
	mut result := ifft2(input)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, true, input.size))
}

// rfft_norm computes a one-dimensional real FFT with an explicit normalization mode.
pub fn rfft_norm[T](input &vtl.Tensor[T], norm FftNorm) !&vtl.Tensor[complex.Complex] {
	mut result := rfft[T](input)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, false, input.size))
}

// irfft_norm reconstructs a real signal with an explicit inverse normalization mode.
pub fn irfft_norm(input &vtl.Tensor[complex.Complex], length int, norm FftNorm) !&vtl.Tensor[f64] {
	mut result := irfft(input, length)!
	return scale_real_fft(mut result, fft_norm_factor(norm, true, length))
}

// rfft_axis_norm computes a selected-axis real FFT with explicit normalization.
pub fn rfft_axis_norm[T](input &vtl.Tensor[T], axis int, norm FftNorm) !&vtl.Tensor[complex.Complex] {
	axis_index := normalize_fft_axis(input.rank(), axis)!
	mut result := rfft_axis[T](input, axis_index)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, false, input.shape[axis_index]))
}

// irfft_axis_norm reconstructs one real tensor axis with explicit normalization.
pub fn irfft_axis_norm(input &vtl.Tensor[complex.Complex], axis int, length int, norm FftNorm) !&vtl.Tensor[f64] {
	mut result := irfft_axis(input, axis, length)!
	return scale_real_fft(mut result, fft_norm_factor(norm, true, length))
}

// rfftn_norm computes a compact real FFT over all input axes with explicit normalization.
pub fn rfftn_norm[T](input &vtl.Tensor[T], norm FftNorm) !&vtl.Tensor[complex.Complex] {
	mut result := rfftn[T](input)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, false, input.size))
}

// irfftn_norm reconstructs a real tensor with explicit N-dimensional normalization.
pub fn irfftn_norm(input &vtl.Tensor[complex.Complex], shape []int, norm FftNorm) !&vtl.Tensor[f64] {
	mut result := irfftn(input, shape)!
	return scale_real_fft(mut result, fft_norm_factor(norm, true, product(shape)))
}

// rfft2_norm computes a compact two-dimensional real FFT with explicit normalization.
pub fn rfft2_norm[T](input &vtl.Tensor[T], norm FftNorm) !&vtl.Tensor[complex.Complex] {
	mut result := rfft2[T](input)!
	return scale_complex_fft(mut result, fft_norm_factor(norm, false, input.size))
}

// irfft2_norm reconstructs a two-dimensional real tensor with explicit normalization.
pub fn irfft2_norm(input &vtl.Tensor[complex.Complex], shape []int, norm FftNorm) !&vtl.Tensor[f64] {
	mut result := irfft2(input, shape)!
	return scale_real_fft(mut result, fft_norm_factor(norm, true, product(shape)))
}

fn fft_norm_factor(norm FftNorm, inverse bool, length int) f64 {
	match norm {
		.backward {
			return 1.0
		}
		.forward {
			return if inverse { f64(length) } else { 1.0 / f64(length) }
		}
		.ortho {
			root := math.sqrt(f64(length))
			return if inverse { root } else { 1.0 / root }
		}
	}
}

fn scale_complex_fft(mut input &vtl.Tensor[complex.Complex], factor f64) !&vtl.Tensor[complex.Complex] {
	if factor == 1 {
		return input
	}
	// All call sites pass freshly allocated FFT output tensors. Scale their
	// contiguous CPU storage in place to avoid a second full-size allocation.
	for i in 0 .. input.size {
		value := input.data.data[i]
		input.data.data[i] = complex.complex(value.re * factor, value.im * factor)
	}
	return input
}

fn scale_real_fft(mut input &vtl.Tensor[f64], factor f64) !&vtl.Tensor[f64] {
	if factor == 1 {
		return input
	}
	// Inverse FFT helpers also return fresh, contiguous CPU output storage.
	for i in 0 .. input.size {
		input.data.data[i] *= factor
	}
	return input
}
