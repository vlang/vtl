// VTL real FFT benchmark. Run from ~/.vmodules with:
// Compile with `-prod` from ~/.vmodules, then run the resulting executable.
module main

import math
import math.complex
import time
import vtl
import vtl.fft

const sizes = [256, 1024, 4096, 16384]
const sample_count = 7

fn main() {
	println('VTL real f64 rfft forward transform')
	println('size,iterations,samples,median_us,min_us,max_us,max_abs_error,checksum')
	for size in sizes {
		iterations := iterations_for(size)
		result := benchmark_rfft(size, iterations) or {
			eprintln('rfft benchmark failed for n=${size}: ${err}')
			exit(1)
		}
		println('${size},${iterations},${sample_count},${result.median_us:.3f},${result.min_us:.3f},${result.max_us:.3f},${result.max_abs_error:.6e},${result.checksum:.6f}')
	}
	println('VTL real f32-input rfft with complex f32 output')
	println('size,iterations,samples,median_us,min_us,max_us,max_abs_error,checksum')
	for size in sizes {
		iterations := iterations_for(size)
		result := benchmark_rfft_f32(size, iterations) or {
			eprintln('f32 rfft benchmark failed for n=${size}: ${err}')
			exit(1)
		}
		println('${size},${iterations},${sample_count},${result.median_us:.3f},${result.min_us:.3f},${result.max_us:.3f},${result.max_abs_error:.6e},${result.checksum:.6f}')
	}
}

struct BenchmarkResult {
	median_us     f64
	min_us        f64
	max_us        f64
	max_abs_error f64
	checksum      f64
}

fn iterations_for(size int) int {
	return match size {
		256 { 100 }
		1024 { 50 }
		4096 { 20 }
		else { 10 }
	}
}

fn benchmark_rfft(size int, iterations int) !BenchmarkResult {
	mut values := []f64{len: size}
	for i in 0 .. size {
		x := f64(i) / f64(size)
		values[i] = math.sin(2.0 * math.pi * 7.0 * x) + 0.25 * math.cos(2.0 * math.pi * 31.0 * x)
	}
	input := vtl.from_1d[f64](values)!
	mut plan := fft.create_rfft_plan[f64](size)!
	defer {
		plan.destroy()
	}
	for _ in 0 .. 3 {
		_ := plan.forward(input)!
	}
	mut timings := []f64{len: sample_count}
	mut result := plan.forward(input)!
	for sample in 0 .. sample_count {
		start := time.sys_mono_now()
		for _ in 0 .. iterations {
			result = plan.forward(input)!
		}
		elapsed_ns := time.sys_mono_now() - start
		timings[sample] = f64(elapsed_ns) / f64(iterations) / 1000.0
	}
	timings.sort()
	return BenchmarkResult{
		median_us:     timings[sample_count / 2]
		min_us:        timings[0]
		max_us:        timings[sample_count - 1]
		max_abs_error: max_abs_rfft_error(size, result)
		checksum:      weighted_rfft_checksum(result)
	}
}

fn benchmark_rfft_f32(size int, iterations int) !BenchmarkResult {
	mut values := []f32{len: size}
	for i in 0 .. size {
		x := f64(i) / f64(size)
		values[i] = f32(math.sin(2.0 * math.pi * 7.0 * x) + 0.25 * math.cos(2.0 * math.pi * 31.0 * x))
	}
	input := vtl.from_1d[f32](values)!
	mut plan := fft.create_rfft_f32_plan(size)!
	defer {
		plan.destroy()
	}
	for _ in 0 .. 3 {
		_ := plan.forward(input)!
	}
	mut timings := []f64{len: sample_count}
	mut result := plan.forward(input)!
	for sample in 0 .. sample_count {
		start := time.sys_mono_now()
		for _ in 0 .. iterations {
			result = plan.forward(input)!
		}
		elapsed_ns := time.sys_mono_now() - start
		timings[sample] = f64(elapsed_ns) / f64(iterations) / 1000.0
	}
	timings.sort()
	return BenchmarkResult{
		median_us:     timings[sample_count / 2]
		min_us:        timings[0]
		max_us:        timings[sample_count - 1]
		max_abs_error: max_abs_rfft_error_f32(size, result)
		checksum:      weighted_rfft_checksum_f32(result)
	}
}

fn max_abs_rfft_error(size int, result &vtl.Tensor[complex.Complex]) f64 {
	mut maximum := 0.0
	for i in 0 .. result.size {
		expected_real := if i == 31 { f64(size) / 8.0 } else { 0.0 }
		expected_imag := if i == 7 { -f64(size) / 2.0 } else { 0.0 }
		value := result.get_nth(i)
		absolute_error := math.hypot(value.re - expected_real, value.im - expected_imag)
		if absolute_error > maximum {
			maximum = absolute_error
		}
	}
	return maximum
}

fn max_abs_rfft_error_f32(size int, result &vtl.Tensor[fft.Complex32]) f64 {
	mut maximum := 0.0
	for i in 0 .. result.size {
		expected_real := if i == 31 { f64(size) / 8.0 } else { 0.0 }
		expected_imag := if i == 7 { -f64(size) / 2.0 } else { 0.0 }
		value := result.get_nth(i)
		absolute_error := math.hypot(f64(value.re) - expected_real, f64(value.im) - expected_imag)
		if absolute_error > maximum {
			maximum = absolute_error
		}
	}
	return maximum
}

fn weighted_rfft_checksum(result &vtl.Tensor[complex.Complex]) f64 {
	mut checksum := 0.0
	for i in 0 .. result.size {
		value := result.get_nth(i)
		weight := f64(i + 1)
		checksum += weight * (value.re + 0.5 * value.im)
	}
	return checksum
}

fn weighted_rfft_checksum_f32(result &vtl.Tensor[fft.Complex32]) f64 {
	mut checksum := 0.0
	for i in 0 .. result.size {
		value := result.get_nth(i)
		weight := f64(i + 1)
		checksum += weight * (f64(value.re) + 0.5 * f64(value.im))
	}
	return checksum
}
