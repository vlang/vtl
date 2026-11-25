// Batched-axis and N-D real FFT benchmark. Run from ~/.vmodules.
module main

import math
import math.complex
import time
import vtl
import vtl.fft

const sample_count = 7

struct BenchmarkResult {
	median_us     f64
	min_us        f64
	max_us        f64
	max_abs_error f64
	checksum      f64
}

fn main() {
	println('transform,shape,iterations,samples,median_us,min_us,max_us,max_abs_error,checksum')
	axis_shape := [64, 512]
	axis_input := make_input(axis_shape)!
	axis_result := benchmark_axis(axis_input, 1, 5)!
	print_result('batched_rfft_axis', '64x512', 5, axis_result)
	axis_input_f32 := make_input_f32(axis_shape)!
	axis_result_f32 := benchmark_axis_f32(axis_input_f32, 1, 5)!
	print_result('batched_rfft_axis_f32', '64x512', 5, axis_result_f32)
	axis_result_f32_generic := benchmark_axis_generic_f32(axis_input_f32, 1, 5)!
	print_result('batched_rfft_axis_f32_generic', '64x512', 5, axis_result_f32_generic)

	nd_shape := [32, 32, 64]
	nd_input := make_input(nd_shape)!
	nd_result := benchmark_nd(nd_input, 3)!
	print_result('rfftn_3d', '32x32x64', 3, nd_result)
	nd_input_f32 := make_input_f32(nd_shape)!
	nd_result_f32 := benchmark_nd_f32(nd_input_f32, 3)!
	print_result('rfftn_3d_f32', '32x32x64', 3, nd_result_f32)
	nd_result_f32_generic := benchmark_nd_generic_f32(nd_input_f32, 3)!
	print_result('rfftn_3d_f32_generic', '32x32x64', 3, nd_result_f32_generic)
}

fn make_input(shape []int) !&vtl.Tensor[f64] {
	mut size := 1
	for dimension in shape {
		size *= dimension
	}
	mut values := []f64{len: size}
	for i in 0 .. size {
		x := f64(i)
		values[i] = math.sin(x * 0.0017) + 0.25 * math.cos(x * 0.0023)
	}
	return vtl.from_array[f64](values, shape)
}

fn make_input_f32(shape []int) !&vtl.Tensor[f32] {
	mut size := 1
	for dimension in shape {
		size *= dimension
	}
	mut values := []f32{len: size}
	for i in 0 .. size {
		x := f64(i)
		values[i] = f32(math.sin(x * 0.0017) + 0.25 * math.cos(x * 0.0023))
	}
	return vtl.from_array[f32](values, shape)
}

fn benchmark_axis(input &vtl.Tensor[f64], axis int, iterations int) !BenchmarkResult {
	for _ in 0 .. 3 {
		_ := fft.rfft_axis(input, axis)!
	}
	mut result := fft.rfft_axis(input, axis)!
	mut timings := []f64{len: sample_count}
	for sample in 0 .. sample_count {
		start := time.sys_mono_now()
		for _ in 0 .. iterations {
			result = fft.rfft_axis(input, axis)!
		}
		elapsed_ns := time.sys_mono_now() - start
		timings[sample] = f64(elapsed_ns) / f64(iterations) / 1000.0
	}
	timings.sort()
	reconstructed := fft.irfft_axis(result, axis, input.shape[axis])!
	return summarize(result, input, reconstructed, timings)
}

fn benchmark_axis_f32(input &vtl.Tensor[f32], axis int, iterations int) !BenchmarkResult {
	for _ in 0 .. 3 {
		_ := fft.rfft_axis_f32(input, axis)!
	}
	mut result := fft.rfft_axis_f32(input, axis)!
	mut timings := []f64{len: sample_count}
	for sample in 0 .. sample_count {
		start := time.sys_mono_now()
		for _ in 0 .. iterations {
			result = fft.rfft_axis_f32(input, axis)!
		}
		elapsed_ns := time.sys_mono_now() - start
		timings[sample] = f64(elapsed_ns) / f64(iterations) / 1000.0
	}
	timings.sort()
	reconstructed := fft.irfft_axis_f32(result, axis, input.shape[axis])!
	return summarize_f32(result, input, reconstructed, timings)
}

fn benchmark_axis_generic_f32(input &vtl.Tensor[f32], axis int, iterations int) !BenchmarkResult {
	for _ in 0 .. 3 {
		_ := fft.rfft_axis[f32](input, axis)!
	}
	mut result := fft.rfft_axis[f32](input, axis)!
	mut timings := []f64{len: sample_count}
	for sample in 0 .. sample_count {
		start := time.sys_mono_now()
		for _ in 0 .. iterations {
			result = fft.rfft_axis[f32](input, axis)!
		}
		elapsed_ns := time.sys_mono_now() - start
		timings[sample] = f64(elapsed_ns) / f64(iterations) / 1000.0
	}
	timings.sort()
	reconstructed := fft.irfft_axis(result, axis, input.shape[axis])!
	return summarize_generic_f32(result, input, reconstructed, timings)
}

fn benchmark_nd(input &vtl.Tensor[f64], iterations int) !BenchmarkResult {
	for _ in 0 .. 3 {
		_ := fft.rfftn(input)!
	}
	mut result := fft.rfftn(input)!
	mut timings := []f64{len: sample_count}
	for sample in 0 .. sample_count {
		start := time.sys_mono_now()
		for _ in 0 .. iterations {
			result = fft.rfftn(input)!
		}
		elapsed_ns := time.sys_mono_now() - start
		timings[sample] = f64(elapsed_ns) / f64(iterations) / 1000.0
	}
	timings.sort()
	reconstructed := fft.irfftn(result, input.shape)!
	return summarize(result, input, reconstructed, timings)
}

fn benchmark_nd_f32(input &vtl.Tensor[f32], iterations int) !BenchmarkResult {
	for _ in 0 .. 3 {
		_ := fft.rfftn_f32(input)!
	}
	mut result := fft.rfftn_f32(input)!
	mut timings := []f64{len: sample_count}
	for sample in 0 .. sample_count {
		start := time.sys_mono_now()
		for _ in 0 .. iterations {
			result = fft.rfftn_f32(input)!
		}
		elapsed_ns := time.sys_mono_now() - start
		timings[sample] = f64(elapsed_ns) / f64(iterations) / 1000.0
	}
	timings.sort()
	reconstructed := fft.irfftn_f32(result, input.shape)!
	return summarize_f32(result, input, reconstructed, timings)
}

fn benchmark_nd_generic_f32(input &vtl.Tensor[f32], iterations int) !BenchmarkResult {
	for _ in 0 .. 3 {
		_ := fft.rfftn[f32](input)!
	}
	mut result := fft.rfftn[f32](input)!
	mut timings := []f64{len: sample_count}
	for sample in 0 .. sample_count {
		start := time.sys_mono_now()
		for _ in 0 .. iterations {
			result = fft.rfftn[f32](input)!
		}
		elapsed_ns := time.sys_mono_now() - start
		timings[sample] = f64(elapsed_ns) / f64(iterations) / 1000.0
	}
	timings.sort()
	reconstructed := fft.irfftn(result, input.shape)!
	return summarize_generic_f32(result, input, reconstructed, timings)
}

fn summarize(
	result &vtl.Tensor[complex.Complex],
	input &vtl.Tensor[f64],
	reconstructed &vtl.Tensor[f64],
	timings []f64
) BenchmarkResult {
	mut max_abs_error := 0.0
	for i in 0 .. input.size {
		max_abs_error = math.max(max_abs_error, math.abs(reconstructed.get_nth(i) - input.get_nth(i)))
	}
	return BenchmarkResult{
		median_us:     timings[sample_count / 2]
		min_us:        timings[0]
		max_us:        timings[sample_count - 1]
		max_abs_error: max_abs_error
		checksum:      weighted_checksum(result)
	}
}

fn summarize_generic_f32(
	result &vtl.Tensor[complex.Complex],
	input &vtl.Tensor[f32],
	reconstructed &vtl.Tensor[f64],
	timings []f64
) BenchmarkResult {
	mut max_abs_error := 0.0
	for i in 0 .. input.size {
		max_abs_error = math.max(max_abs_error, math.abs(reconstructed.get_nth(i) - f64(input.get_nth(i))))
	}
	return BenchmarkResult{
		median_us:     timings[sample_count / 2]
		min_us:        timings[0]
		max_us:        timings[sample_count - 1]
		max_abs_error: max_abs_error
		checksum:      weighted_checksum(result)
	}
}

fn summarize_f32(
	result &vtl.Tensor[fft.Complex32],
	input &vtl.Tensor[f32],
	reconstructed &vtl.Tensor[f32],
	timings []f64
) BenchmarkResult {
	mut max_abs_error := 0.0
	for i in 0 .. input.size {
		max_abs_error = math.max(max_abs_error, math.abs(f64(reconstructed.get_nth(i) - input.get_nth(i))))
	}
	return BenchmarkResult{
		median_us:     timings[sample_count / 2]
		min_us:        timings[0]
		max_us:        timings[sample_count - 1]
		max_abs_error: max_abs_error
		checksum:      weighted_checksum_f32(result)
	}
}

fn weighted_checksum(result &vtl.Tensor[complex.Complex]) f64 {
	mut checksum := 0.0
	for i in 0 .. result.size {
		value := result.get_nth(i)
		checksum += f64(i + 1) * (value.re + 0.5 * value.im)
	}
	return checksum
}

fn weighted_checksum_f32(result &vtl.Tensor[fft.Complex32]) f64 {
	mut checksum := 0.0
	for i in 0 .. result.size {
		value := result.get_nth(i)
		checksum += f64(i + 1) * (f64(value.re) + 0.5 * f64(value.im))
	}
	return checksum
}

fn print_result(name string, shape string, iterations int, result BenchmarkResult) {
	println('${name},${shape},${iterations},${sample_count},${result.median_us:.3f},${result.min_us:.3f},${result.max_us:.3f},${result.max_abs_error:.6e},${result.checksum:.6f}')
}
