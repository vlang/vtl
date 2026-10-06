module fft

import vsl.fft as vsl_fft
import vtl

// RealFftPlanF32 stores a reusable native plan for one real f32 vector length.
pub struct RealFftPlanF32 {
pub:
	length int
mut:
	plan      vsl_fft.Fftplan
	destroyed bool
	scratch   []f32
}

// create_rfft_f32_plan creates a reusable real f32 FFT plan that returns
// Complex32 output without promoting values to f64.
pub fn create_rfft_f32_plan(length int) !RealFftPlanF32 {
	if length <= 0 {
		return error('f32 FFT plan length must be positive')
	}
	scratch := []f32{len: length}
	plan := vsl_fft.create_plan(scratch) or { return error('could not create an f32 FFT plan') }
	return RealFftPlanF32{
		length:  length
		plan:    plan
		scratch: scratch
	}
}

// forward computes a compact real f32 FFT, reusing the native plan and scratch.
// Calls on one plan are mutable and must not run concurrently.
pub fn (mut plan RealFftPlanF32) forward(input &vtl.Tensor[f32]) !&vtl.Tensor[Complex32] {
	if plan.destroyed {
		return error('f32 FFT plan has been destroyed')
	}
	if input.rank() != 1 || input.size != plan.length {
		return error('f32 FFT input must be a vector of length ${plan.length}')
	}
	copy_real_f32(input, mut plan.scratch)
	if vsl_fft.forward_fft(plan.plan, mut plan.scratch) != 0 {
		return error('f32 FFT backend failed to compute the forward transform')
	}
	return unpack_rfft_complex32(plan.scratch)
}

// destroy releases the native plan. Repeated calls are safe.
pub fn (mut plan RealFftPlanF32) destroy() {
	if !plan.destroyed {
		vsl_fft.destroy_plan(plan.plan)
		plan.destroyed = true
	}
}

// rfft_f32 computes a one-dimensional real f32 transform and returns compact
// f32 complex bins without promoting the spectrum to f64.
pub fn rfft_f32(input &vtl.Tensor[f32]) !&vtl.Tensor[Complex32] {
	if input.rank() != 1 || input.size == 0 {
		return error('rfft_f32 expects a non-empty one-dimensional tensor')
	}
	mut packed := []f32{len: input.size}
	plan := vsl_fft.create_plan(packed) or { return error('rfft_f32 could not create an FFT plan') }
	defer {
		vsl_fft.destroy_plan(plan)
	}
	copy_real_f32(input, mut packed)
	if vsl_fft.forward_fft(plan, mut packed) != 0 {
		return error('rfft_f32 backend failed to compute the forward transform')
	}
	return unpack_rfft_complex32(packed)
}

// rfft_axis_f32 computes a compact real f32 FFT along one selected axis.
pub fn rfft_axis_f32(input &vtl.Tensor[f32], axis int) !&vtl.Tensor[Complex32] {
	axis_index := normalize_fft_axis(input.rank(), axis)!
	if input.size == 0 {
		return error('rfft_axis_f32 requires a non-empty tensor')
	}
	axis_length := input.shape[axis_index]
	frequency_count := axis_length / 2 + 1
	mut output_shape := input.shape.clone()
	output_shape[axis_index] = frequency_count
	mut output := []Complex32{len: product(output_shape)}
	mut line := []f32{len: axis_length}
	mut index := []int{len: input.rank()}
	plan := vsl_fft.create_plan(line) or { return error('rfft_axis_f32 could not create an FFT plan') }
	defer {
		vsl_fft.destroy_plan(plan)
	}
	line_count := input.size / axis_length
	for line_index in 0 .. line_count {
		decode_fft_axis_line(line_index, input.shape, axis_index, mut index)
		for position in 0 .. axis_length {
			index[axis_index] = position
			line[position] = input.get(index)
		}
		if vsl_fft.forward_fft(plan, mut line) != 0 {
			return error('rfft_axis_f32 backend failed to compute the forward transform')
		}
		write_rfft_complex32_line(line, axis_length, axis_index, mut index, output_shape, mut output)
	}
	return tensor_from_owned[Complex32](output, output_shape)
}

// rfftn_f32 computes a compact real f32 FFT over all tensor axes.
pub fn rfftn_f32(input &vtl.Tensor[f32]) !&vtl.Tensor[Complex32] {
	if input.rank() == 0 || input.size == 0 {
		return error('rfftn_f32 expects a non-empty tensor with at least one dimension')
	}
	last_axis := input.rank() - 1
	mut result := rfft_axis_f32(input, last_axis)!
	for axis in 0 .. last_axis {
		result = transform_complex_axis_f32(result, axis, false)!
	}
	return result
}

// rfft2_f32 computes a compact real f32 FFT for a two-dimensional tensor.
pub fn rfft2_f32(input &vtl.Tensor[f32]) !&vtl.Tensor[Complex32] {
	if input.rank() != 2 {
		return error('rfft2_f32 expects a two-dimensional tensor')
	}
	return rfftn_f32(input)
}

// irfft_f32 reconstructs a real f32 vector from compact f32 complex bins.
pub fn irfft_f32(input &vtl.Tensor[Complex32], length int) !&vtl.Tensor[f32] {
	if input.rank() != 1 || input.size == 0 {
		return error('irfft_f32 expects a non-empty one-dimensional spectrum')
	}
	if length <= 0 || input.size != length / 2 + 1 {
		return error('irfft_f32 spectrum size does not match output length ${length}')
	}
	mut packed := []f32{len: length}
	mut index := [0]
	pack_irfft_complex32_line(input, length, 0, mut packed, mut index)
	plan := vsl_fft.create_plan(packed) or { return error('irfft_f32 could not create an FFT plan') }
	defer {
		vsl_fft.destroy_plan(plan)
	}
	if vsl_fft.backward_fft(plan, mut packed) != 0 {
		return error('irfft_f32 backend failed to compute the inverse transform')
	}
	for i in 0 .. packed.len {
		packed[i] /= f32(length)
	}
	return tensor_from_owned[f32](packed, [length])
}

// irfft_axis_f32 reconstructs one real f32 tensor axis from compact bins.
pub fn irfft_axis_f32(input &vtl.Tensor[Complex32], axis int, length int) !&vtl.Tensor[f32] {
	axis_index := normalize_fft_axis(input.rank(), axis)!
	if length <= 0 || input.size == 0 {
		return error('irfft_axis_f32 requires a positive length and non-empty spectrum')
	}
	if input.shape[axis_index] != length / 2 + 1 {
		return error('irfft_axis_f32 spectrum shape does not match output length ${length}')
	}
	mut output_shape := input.shape.clone()
	output_shape[axis_index] = length
	mut output := []f32{len: product(output_shape)}
	mut line := []f32{len: length}
	mut index := []int{len: input.rank()}
	plan := vsl_fft.create_plan(line) or { return error('irfft_axis_f32 could not create an FFT plan') }
	defer {
		vsl_fft.destroy_plan(plan)
	}
	for line_index in 0 .. output.len / length {
		decode_fft_axis_line(line_index, output_shape, axis_index, mut index)
		pack_irfft_complex32_line(input, length, axis_index, mut line, mut index)
		if vsl_fft.backward_fft(plan, mut line) != 0 {
			return error('irfft_axis_f32 backend failed to compute the inverse transform')
		}
		for position in 0 .. length {
			index[axis_index] = position
			output[row_major_index(index, output_shape)] = line[position] / f32(length)
		}
	}
	return tensor_from_owned[f32](output, output_shape)
}

// irfftn_f32 reconstructs a real f32 tensor from its compact N-D spectrum.
pub fn irfftn_f32(input &vtl.Tensor[Complex32], shape []int) !&vtl.Tensor[f32] {
	if shape.len == 0 || input.rank() != shape.len || input.size == 0 {
		return error('irfftn_f32 expects a non-empty spectrum and matching shape')
	}
	for axis, dimension in shape {
		if dimension <= 0 {
			return error('irfftn_f32 dimensions must be positive')
		}
		expected := if axis == shape.len - 1 { dimension / 2 + 1 } else { dimension }
		if input.shape[axis] != expected {
			return error('irfftn_f32 spectrum shape does not match requested output shape')
		}
	}
	last_axis := shape.len - 1
	mut transformed := &vtl.Tensor[Complex32](unsafe { nil })
	if last_axis == 0 {
		transformed = input
	} else {
		transformed = transform_complex_axis_f32(input, 0, true)!
		for axis in 1 .. last_axis {
			transformed = transform_complex_axis_f32(transformed, axis, true)!
		}
	}
	last_length := shape[last_axis]
	mut output := []f32{len: product(shape)}
	mut line := []f32{len: last_length}
	mut index := []int{len: shape.len}
	plan := vsl_fft.create_plan(line) or { return error('irfftn_f32 could not create an FFT plan') }
	defer {
		vsl_fft.destroy_plan(plan)
	}
	for line_index in 0 .. output.len / last_length {
		decode_row_major_index(line_index, shape[..last_axis], mut index)
		pack_irfft_complex32_line(transformed, last_length, last_axis, mut line, mut index)
		if vsl_fft.backward_fft(plan, mut line) != 0 {
			return error('irfftn_f32 backend failed to compute the inverse transform')
		}
		for position in 0 .. last_length {
			index[last_axis] = position
			output[row_major_index(index, shape)] = line[position] / f32(last_length)
		}
	}
	return tensor_from_owned[f32](output, shape)
}

// irfft2_f32 reconstructs a two-dimensional f32 tensor from compact bins.
pub fn irfft2_f32(input &vtl.Tensor[Complex32], shape []int) !&vtl.Tensor[f32] {
	if shape.len != 2 {
		return error('irfft2_f32 expects a two-dimensional output shape')
	}
	return irfftn_f32(input, shape)
}

fn copy_real_f32(input &vtl.Tensor[f32], mut packed []f32) {
	if input.is_row_major_contiguous() {
		unsafe { C.memcpy(packed.data, input.data.data.data, input.size * sizeof(f32)) }
	} else {
		for i in 0 .. input.size {
			packed[i] = input.get_nth(i)
		}
	}
}

fn unpack_rfft_complex32(packed []f32) !&vtl.Tensor[Complex32] {
	mut frequencies := []Complex32{len: packed.len / 2 + 1}
	frequencies[0] = Complex32{ re: packed[0], im: 0 }
	for frequency in 1 .. (packed.len + 1) / 2 {
		frequencies[frequency] = Complex32{
			re: packed[2 * frequency - 1]
			im: packed[2 * frequency]
		}
	}
	if packed.len % 2 == 0 && packed.len > 1 {
		frequencies[packed.len / 2] = Complex32{ re: packed[packed.len - 1], im: 0 }
	}
	return tensor_from_owned[Complex32](frequencies, [frequencies.len])
}

fn write_rfft_complex32_line(line []f32, length int, axis int, mut index []int, shape []int, mut output []Complex32) {
	frequency_count := length / 2 + 1
	for frequency in 0 .. frequency_count {
		index[axis] = frequency
		output_index := row_major_index(index, shape)
		if frequency == 0 {
			output[output_index] = Complex32{ re: line[0], im: 0 }
		} else if length % 2 == 0 && frequency == length / 2 {
			output[output_index] = Complex32{ re: line[length - 1], im: 0 }
		} else {
			output[output_index] = Complex32{
				re: line[2 * frequency - 1]
				im: line[2 * frequency]
			}
		}
	}
}

fn pack_irfft_complex32_line(input &vtl.Tensor[Complex32], length int, axis int, mut line []f32, mut index []int) {
	index[axis] = 0
	line[0] = input.get(index).re
	for frequency in 1 .. (length + 1) / 2 {
		index[axis] = frequency
		value := input.get(index)
		line[2 * frequency - 1] = value.re
		line[2 * frequency] = value.im
	}
	if length % 2 == 0 && length > 1 {
		index[axis] = length / 2
		line[length - 1] = input.get(index).re
	}
}
