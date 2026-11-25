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

	nd_shape := [32, 32, 64]
	nd_input := make_input(nd_shape)!
	nd_result := benchmark_nd(nd_input, 3)!
	print_result('rfftn_3d', '32x32x64', 3, nd_result)
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

fn weighted_checksum(result &vtl.Tensor[complex.Complex]) f64 {
	mut checksum := 0.0
	for i in 0 .. result.size {
		value := result.get_nth(i)
		checksum += f64(i + 1) * (value.re + 0.5 * value.im)
	}
	return checksum
}

fn print_result(name string, shape string, iterations int, result BenchmarkResult) {
	println('${name},${shape},${iterations},${sample_count},${result.median_us:.3f},${result.min_us:.3f},${result.max_us:.3f},${result.max_abs_error:.6e},${result.checksum:.6f}')
}
