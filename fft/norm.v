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

// fft_norm_f32 computes a one-dimensional complex f32 FFT with explicit normalization.
pub fn fft_norm_f32(input &vtl.Tensor[Complex32], norm FftNorm) !&vtl.Tensor[Complex32] {
	mut result := fft_f32(input)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, false, input.size))
}

// ifft_norm_f32 computes a normalized one-dimensional inverse complex f32 FFT.
pub fn ifft_norm_f32(input &vtl.Tensor[Complex32], norm FftNorm) !&vtl.Tensor[Complex32] {
	mut result := ifft_f32(input)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, true, input.size))
}

// fft_axis_norm_f32 transforms one complex f32 tensor axis with explicit normalization.
pub fn fft_axis_norm_f32(input &vtl.Tensor[Complex32], axis int, norm FftNorm) !&vtl.Tensor[Complex32] {
	axis_index := normalize_fft_axis(input.rank(), axis)!
	mut result := fft_axis_f32(input, axis_index)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, false, input.shape[axis_index]))
}

// ifft_axis_norm_f32 applies explicit inverse normalization on one complex f32 axis.
pub fn ifft_axis_norm_f32(input &vtl.Tensor[Complex32], axis int, norm FftNorm) !&vtl.Tensor[Complex32] {
	axis_index := normalize_fft_axis(input.rank(), axis)!
	mut result := ifft_axis_f32(input, axis_index)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, true, input.shape[axis_index]))
}

// fftn_norm_f32 transforms all complex f32 axes with explicit normalization.
pub fn fftn_norm_f32(input &vtl.Tensor[Complex32], norm FftNorm) !&vtl.Tensor[Complex32] {
	mut result := fftn_f32(input)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, false, input.size))
}

// ifftn_norm_f32 applies explicit inverse normalization across all complex f32 axes.
pub fn ifftn_norm_f32(input &vtl.Tensor[Complex32], norm FftNorm) !&vtl.Tensor[Complex32] {
	mut result := ifftn_f32(input)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, true, input.size))
}

// fft2_norm_f32 computes a 2-D complex f32 FFT with explicit normalization.
pub fn fft2_norm_f32(input &vtl.Tensor[Complex32], norm FftNorm) !&vtl.Tensor[Complex32] {
	mut result := fft2_f32(input)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, false, input.size))
}

// ifft2_norm_f32 computes a 2-D complex f32 inverse FFT with explicit normalization.
pub fn ifft2_norm_f32(input &vtl.Tensor[Complex32], norm FftNorm) !&vtl.Tensor[Complex32] {
	mut result := ifft2_f32(input)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, true, input.size))
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

// rfft_norm_f32 computes a one-dimensional real f32 transform with normalization.
pub fn rfft_norm_f32(input &vtl.Tensor[f32], norm FftNorm) !&vtl.Tensor[Complex32] {
	mut result := rfft_f32(input)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, false, input.size))
}

// irfft_norm_f32 reconstructs a real f32 signal with explicit inverse normalization.
pub fn irfft_norm_f32(input &vtl.Tensor[Complex32], length int, norm FftNorm) !&vtl.Tensor[f32] {
	mut result := irfft_f32(input, length)!
	return scale_real_fft_f32(mut result, fft_norm_factor(norm, true, length))
}

// rfft_axis_norm_f32 computes a selected-axis real f32 FFT with normalization.
pub fn rfft_axis_norm_f32(input &vtl.Tensor[f32], axis int, norm FftNorm) !&vtl.Tensor[Complex32] {
	axis_index := normalize_fft_axis(input.rank(), axis)!
	mut result := rfft_axis_f32(input, axis_index)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, false, input.shape[axis_index]))
}

// irfft_axis_norm_f32 applies inverse normalization when reconstructing one f32 axis.
pub fn irfft_axis_norm_f32(input &vtl.Tensor[Complex32], axis int, length int, norm FftNorm) !&vtl.Tensor[f32] {
	mut result := irfft_axis_f32(input, axis, length)!
	return scale_real_fft_f32(mut result, fft_norm_factor(norm, true, length))
}

// rfftn_norm_f32 computes a normalized real f32 transform over all axes.
pub fn rfftn_norm_f32(input &vtl.Tensor[f32], norm FftNorm) !&vtl.Tensor[Complex32] {
	mut result := rfftn_f32(input)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, false, input.size))
}

// irfftn_norm_f32 reconstructs a real f32 tensor with explicit N-D normalization.
pub fn irfftn_norm_f32(input &vtl.Tensor[Complex32], shape []int, norm FftNorm) !&vtl.Tensor[f32] {
	mut result := irfftn_f32(input, shape)!
	return scale_real_fft_f32(mut result, fft_norm_factor(norm, true, product(shape)))
}

// rfft2_norm_f32 computes a normalized two-dimensional real f32 transform.
pub fn rfft2_norm_f32(input &vtl.Tensor[f32], norm FftNorm) !&vtl.Tensor[Complex32] {
	mut result := rfft2_f32(input)!
	return scale_complex_fft_f32(mut result, fft_norm_factor(norm, false, input.size))
}

// irfft2_norm_f32 reconstructs a real f32 tensor with explicit 2-D normalization.
pub fn irfft2_norm_f32(input &vtl.Tensor[Complex32], shape []int, norm FftNorm) !&vtl.Tensor[f32] {
	mut result := irfft2_f32(input, shape)!
	return scale_real_fft_f32(mut result, fft_norm_factor(norm, true, product(shape)))
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

fn scale_complex_fft_f32(mut input &vtl.Tensor[Complex32], factor f64) !&vtl.Tensor[Complex32] {
	if factor == 1 {
		return input
	}
	factor_f32 := f32(factor)
	for i in 0 .. input.size {
		value := input.data.data[i]
		input.data.data[i] = Complex32{
			re: value.re * factor_f32
			im: value.im * factor_f32
		}
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

fn scale_real_fft_f32(mut input &vtl.Tensor[f32], factor f64) !&vtl.Tensor[f32] {
	if factor == 1 {
		return input
	}
	factor_f32 := f32(factor)
	for i in 0 .. input.size {
		input.data.data[i] *= factor_f32
	}
	return input
}
