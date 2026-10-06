module fft

import math.complex
import math
import vsl.fft as vsl_fft
import vtl
import vtl.storage

// fftfreq returns the discrete Fourier Transform sample frequencies for a
// transform of length n and sample spacing d, in NumPy's native FFT order.
pub fn fftfreq(n int, d f64) !&vtl.Tensor[f64] {
	validate_frequency_args(n, d, 'fftfreq')!
	step := 1.0 / (f64(n) * d)
	mut frequencies := []f64{len: n}
	positive_end := (n + 1) / 2
	for index in 0 .. positive_end {
		frequencies[index] = f64(index) * step
	}
	for index in positive_end .. n {
		frequencies[index] = f64(index - n) * step
	}
	return vtl.from_1d(frequencies)
}

// rfftfreq returns the non-negative discrete Fourier Transform sample
// frequencies for a real-input transform of length n and sample spacing d.
pub fn rfftfreq(n int, d f64) !&vtl.Tensor[f64] {
	validate_frequency_args(n, d, 'rfftfreq')!
	step := 1.0 / (f64(n) * d)
	mut frequencies := []f64{len: n / 2 + 1}
	for index in 0 .. frequencies.len {
		frequencies[index] = f64(index) * step
	}
	return vtl.from_1d(frequencies)
}

fn validate_frequency_args(n int, d f64, operation string) ! {
	if n <= 0 {
		return error('${operation} requires a positive transform length')
	}
	if d == 0 || math.is_nan(d) || math.is_inf(d, 0) {
		return error('${operation} requires a finite non-zero sample spacing')
	}
}

// fftshift moves the zero-frequency component to the center of every axis.
pub fn fftshift[T](input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	mut axes := []int{cap: input.rank()}
	for axis in 0 .. input.rank() {
		axes << axis
	}
	return shift_axes[T](input, axes, false)
}

// ifftshift moves the zero-frequency component back to the start of every
// axis. It is the inverse of fftshift for odd and even dimensions.
pub fn ifftshift[T](input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	mut axes := []int{cap: input.rank()}
	for axis in 0 .. input.rank() {
		axes << axis
	}
	return shift_axes[T](input, axes, true)
}

// fftshift_axis shifts one axis so its zero-frequency component is centered.
pub fn fftshift_axis[T](input &vtl.Tensor[T], axis int) !&vtl.Tensor[T] {
	axis_index := normalize_shift_axis(input, axis, 'fftshift_axis')!
	return shift_axes[T](input, [axis_index], false)
}

// ifftshift_axis reverses fftshift_axis for one axis.
pub fn ifftshift_axis[T](input &vtl.Tensor[T], axis int) !&vtl.Tensor[T] {
	axis_index := normalize_shift_axis(input, axis, 'ifftshift_axis')!
	return shift_axes[T](input, [axis_index], true)
}

fn normalize_shift_axis[T](input &vtl.Tensor[T], axis int, operation string) !int {
	rank := input.rank()
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('${operation} axis ${axis} is out of bounds for rank ${rank}')
	}
	return axis_index
}

fn shift_axes[T](input &vtl.Tensor[T], axes []int, inverse bool) !&vtl.Tensor[T] {
	mut output := []T{len: input.size}
	mut input_index := []int{len: input.rank()}
	for flat_index in 0 .. input.size {
		decode_row_major_index(flat_index, input.shape, mut input_index)
		for axis in axes {
			length := input.shape[axis]
			if length > 0 {
				shift := if inverse { length / 2 } else { (length + 1) / 2 }
				input_index[axis] = (input_index[axis] + shift) % length
			}
		}
		output[flat_index] = input.get(input_index)
	}
	return tensor_from_owned[T](output, input.shape)
}

// RealFftPlan stores a reusable PocketFFT plan for one real input length.
// Call destroy when finished to release the native backend plan.
pub struct RealFftPlan[T] {
pub:
	length int
mut:
	plan        vsl_fft.Fftplan
	destroyed   bool
	scratch_f32 []f32
	scratch_f64 []f64
}

// create_rfft_plan creates a reusable one-dimensional real FFT plan.
pub fn create_rfft_plan[T](length int) !RealFftPlan[T] {
	if length <= 0 {
		return error('FFT plan length must be positive')
	}
	$if T is f32 {
		plan := vsl_fft.create_plan([]f32{len: length}) or {
			return error('could not create an f32 FFT plan')
		}
		return RealFftPlan[T]{
			length:      length
			plan:        plan
			scratch_f32: []f32{len: length}
		}
	} $else $if T is f64 {
		plan := vsl_fft.create_plan([]f64{len: length}) or {
			return error('could not create an f64 FFT plan')
		}
		return RealFftPlan[T]{
			length:      length
			plan:        plan
			scratch_f64: []f64{len: length}
		}
	} $else {
		return error('rfft supports f32 and f64 input tensors')
	}
}

// forward computes a compact real FFT using the plan. The input length must
// match the plan length; a new output tensor is allocated for each call. The
// plan reuses an internal mutable work buffer, so calls on one plan must not run
// concurrently.
pub fn (mut plan RealFftPlan[T]) forward(input &vtl.Tensor[T]) !&vtl.Tensor[complex.Complex] {
	if plan.destroyed {
		return error('FFT plan has been destroyed')
	}
	if input.rank() != 1 || input.size != plan.length {
		return error('FFT input must be a vector of length ${plan.length}')
	}
	$if T is f32 {
		return rfft_f32_with_plan(unsafe { &vtl.Tensor[f32](input) }, plan.plan, mut plan.scratch_f32)
	} $else $if T is f64 {
		return rfft_f64_with_plan(unsafe { &vtl.Tensor[f64](input) }, plan.plan, mut plan.scratch_f64)
	} $else {
		return error('rfft supports f32 and f64 input tensors')
	}
}

// destroy releases the native PocketFFT plan. Repeated calls are safe.
pub fn (mut plan RealFftPlan[T]) destroy() {
	if !plan.destroyed {
		vsl_fft.destroy_plan(plan.plan)
		plan.destroyed = true
	}
}

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
		mut plan := create_rfft_plan[f32](input.size)!
		defer {
			plan.destroy()
		}
		return plan.forward(unsafe { &vtl.Tensor[f32](input) })
	} $else $if T is f64 {
		mut plan := create_rfft_plan[f64](input.size)!
		defer {
			plan.destroy()
		}
		return plan.forward(unsafe { &vtl.Tensor[f64](input) })
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

// fft_axis transforms one axis of a complex tensor and preserves its shape.
// Negative axis values count backward from the final dimension.
pub fn fft_axis(input &vtl.Tensor[complex.Complex], axis int) !&vtl.Tensor[complex.Complex] {
	normalized_axis := normalize_fft_axis(input.rank(), axis)!
	return transform_complex_axis(input, normalized_axis, false)
}

// ifft_axis computes a normalized inverse complex transform along one axis.
pub fn ifft_axis(input &vtl.Tensor[complex.Complex], axis int) !&vtl.Tensor[complex.Complex] {
	normalized_axis := normalize_fft_axis(input.rank(), axis)!
	return transform_complex_axis(input, normalized_axis, true)
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

// rfftn computes a real-input Fourier transform across every tensor axis.
// The final axis is stored compactly, with length floor(n/2)+1.
pub fn rfftn[T](input &vtl.Tensor[T]) !&vtl.Tensor[complex.Complex] {
	if input.rank() == 0 || input.size == 0 {
		return error('rfftn expects a non-empty tensor with at least one dimension')
	}
	$if T is f32 || T is f64 {
		last_axis := input.rank() - 1
		last_length := input.shape[last_axis]
		frequency_count := last_length / 2 + 1
		mut output_shape := input.shape.clone()
		output_shape[last_axis] = frequency_count
		mut compact := []complex.Complex{len: product(output_shape)}
		mut index := []int{len: input.rank()}
		mut line := []T{len: last_length}
		plan := vsl_fft.create_plan(line) or { return error('rfftn could not create an FFT plan') }
		defer {
			vsl_fft.destroy_plan(plan)
		}
		line_count := input.size / last_length
		for line_index in 0 .. line_count {
			decode_row_major_index(line_index, input.shape[..last_axis], mut index)
			for position in 0 .. last_length {
				index[last_axis] = position
				line[position] = T(input.get(index))
			}
			if vsl_fft.forward_fft(plan, mut line) != 0 {
				return error('rfftn backend failed to compute the forward transform')
			}
			for frequency in 0 .. frequency_count {
				index[last_axis] = frequency
				output_index := row_major_index(index, output_shape)
				if frequency == 0 {
					compact[output_index] = complex.complex(f64(line[0]), 0)
				} else if last_length % 2 == 0 && frequency == last_length / 2 {
					compact[output_index] = complex.complex(f64(line[last_length - 1]), 0)
				} else {
					compact[output_index] = complex.complex(f64(line[2 * frequency - 1]),
						f64(line[2 * frequency]))
				}
			}
		}
		mut result := tensor_from_owned[complex.Complex](compact, output_shape)!
		for axis in 0 .. last_axis {
			result = transform_complex_axis(result, axis, false)!
		}
		return result
	} $else {
		return error('rfftn supports f32 and f64 input tensors')
	}
}

// rfft2 computes a real-input two-dimensional Fourier transform.
pub fn rfft2[T](input &vtl.Tensor[T]) !&vtl.Tensor[complex.Complex] {
	if input.rank() != 2 {
		return error('rfft2 expects a two-dimensional tensor')
	}
	return rfftn[T](input)
}

// irfftn reconstructs a real tensor from a compact multidimensional spectrum.
// The original shape is required to disambiguate odd and even final axes.
pub fn irfftn(input &vtl.Tensor[complex.Complex], shape []int) !&vtl.Tensor[f64] {
	if shape.len == 0 || input.rank() != shape.len || input.size == 0 {
		return error('irfftn expects a non-empty spectrum and a matching non-empty shape')
	}
	for axis, dimension in shape {
		if dimension <= 0 {
			return error('irfftn dimensions must be positive')
		}
		expected := if axis == shape.len - 1 { dimension / 2 + 1 } else { dimension }
		if input.shape[axis] != expected {
			return error('irfftn spectrum shape does not match requested output shape')
		}
	}
	last_axis := shape.len - 1
	last_length := shape[last_axis]
	mut transformed := &vtl.Tensor[complex.Complex](unsafe { nil })
	if last_axis == 0 {
		transformed = input
	} else {
		transformed = transform_complex_axis(input, 0, true)!
		for axis in 1 .. last_axis {
			transformed = transform_complex_axis(transformed, axis, true)!
		}
	}
	mut output := []f64{len: product(shape)}
	mut line := []f64{len: last_length}
	mut index := []int{len: shape.len}
	plan := vsl_fft.create_plan(line) or { return error('irfftn could not create an FFT plan') }
	defer {
		vsl_fft.destroy_plan(plan)
	}
	line_count := output.len / last_length
	for line_index in 0 .. line_count {
		decode_row_major_index(line_index, shape[..last_axis], mut index)
		index[last_axis] = 0
		line[0] = transformed.get(index).re
		for frequency in 1 .. (last_length + 1) / 2 {
			index[last_axis] = frequency
			value := transformed.get(index)
			line[2 * frequency - 1] = value.re
			line[2 * frequency] = value.im
		}
		if last_length % 2 == 0 && last_length > 1 {
			index[last_axis] = last_length / 2
			line[last_length - 1] = transformed.get(index).re
		}
		if vsl_fft.backward_fft(plan, mut line) != 0 {
			return error('irfftn backend failed to compute the inverse transform')
		}
		for position in 0 .. last_length {
			index[last_axis] = position
			output[row_major_index(index, shape)] = line[position] / f64(last_length)
		}
	}
	return tensor_from_owned[f64](output, shape)
}

// irfft2 reconstructs a two-dimensional real tensor from rfft2 output.
pub fn irfft2(input &vtl.Tensor[complex.Complex], shape []int) !&vtl.Tensor[f64] {
	if shape.len != 2 {
		return error('irfft2 expects a two-dimensional output shape')
	}
	return irfftn(input, shape)
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
	return tensor_from_owned[f64](packed, [packed.len])
}

fn rfft_f32_with_plan(input &vtl.Tensor[f32], plan vsl_fft.Fftplan, mut packed []f32) !&vtl.Tensor[complex.Complex] {
	if input.is_row_major_contiguous() {
		unsafe { C.memcpy(packed.data, input.data.data.data, input.size * sizeof(f32)) }
	} else {
		for i in 0 .. input.size {
			packed[i] = input.get_nth(i)
		}
	}
	if vsl_fft.forward_fft(plan, mut packed) != 0 {
		return error('rfft backend failed to compute the forward transform')
	}
	return unpack_rfft_f32(packed)
}

fn rfft_f64_with_plan(input &vtl.Tensor[f64], plan vsl_fft.Fftplan, mut packed []f64) !&vtl.Tensor[complex.Complex] {
	if input.is_row_major_contiguous() {
		unsafe { C.memcpy(packed.data, input.data.data.data, input.size * sizeof(f64)) }
	} else {
		for i in 0 .. input.size {
			packed[i] = input.get_nth(i)
		}
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

fn normalize_fft_axis(rank int, axis int) !int {
	if rank == 0 {
		return error('FFT axis requires a tensor with at least one dimension')
	}
	normalized_axis := if axis < 0 { axis + rank } else { axis }
	if normalized_axis < 0 || normalized_axis >= rank {
		return error('FFT axis ${axis} out of bounds for rank ${rank}')
	}
	return normalized_axis
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
	return tensor_from_owned[complex.Complex](output, input.shape)
}

fn tensor_from_owned[T](values []T, shape []int) !&vtl.Tensor[T] {
	if product(shape) != values.len {
		return error('FFT output data length does not match its shape')
	}
	mut strides := []int{len: shape.len}
	mut stride := 1
	for axis := shape.len - 1; axis >= 0; axis-- {
		strides[axis] = stride
		stride *= shape[axis]
	}
	return &vtl.Tensor[T]{
		data:    &storage.CpuStorage[T]{
			data: values
		}
		memory:  .row_major
		size:    values.len
		shape:   shape.clone()
		strides: strides
	}
}

fn row_major_index(index []int, shape []int) int {
	mut flat_index := 0
	for axis in 0 .. shape.len {
		flat_index = flat_index * shape[axis] + index[axis]
	}
	return flat_index
}

fn decode_row_major_index(flat_index int, shape []int, mut index []int) {
	mut remainder := flat_index
	for axis := shape.len - 1; axis >= 0; axis-- {
		index[axis] = remainder % shape[axis]
		remainder /= shape[axis]
	}
}

fn product(shape []int) int {
	mut result := 1
	for dimension in shape {
		result *= dimension
	}
	return result
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
	return tensor_from_owned[complex.Complex](frequencies, [frequencies.len])
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
	return tensor_from_owned[complex.Complex](frequencies, [frequencies.len])
}
