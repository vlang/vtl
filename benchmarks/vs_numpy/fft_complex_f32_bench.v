// VTL complex f32 FFT benchmark. Run from ~/.vmodules with:
// Compile with `-prod` from ~/.vmodules, then run the resulting executable.
module main

import math
import time
import vtl
import vtl.fft

const sizes = [256, 1024, 4096, 16384]
const sample_count = 7

fn main() {
	println('VTL complex f32 FFT forward transform')
	println('layout,size,iterations,samples,median_us,min_us,max_us,max_abs_error,checksum')
	for strided in [false, true] {
		layout := if strided { 'strided' } else { 'contiguous' }
		for size in sizes {
			iterations := if size <= 1024 {
				50
			} else if size <= 4096 {
				20
			} else {
				10
			}
			result := benchmark_fft_f32(size, iterations, strided) or {
				eprintln('complex f32 FFT benchmark failed for n=${size}: ${err}')
				exit(1)
			}
			println('${layout},${size},${iterations},${sample_count},${result.median_us:.3f},${result.min_us:.3f},${result.max_us:.3f},${result.max_abs_error:.6e},${result.checksum:.6f}')
		}
	}
}

struct BenchmarkResult {
	median_us     f64
	min_us        f64
	max_us        f64
	max_abs_error f64
	checksum      f64
}

fn benchmark_fft_f32(size int, iterations int, strided bool) !BenchmarkResult {
	input_size := if strided { size * 2 } else { size }
	mut values := []fft.Complex32{len: input_size}
	for i in 0 .. size {
		x := f64(i) / f64(size)
		offset := if strided { i * 2 } else { i }
		values[offset] = fft.Complex32{
			re: f32(math.sin(2.0 * math.pi * 7.0 * x))
			im: f32(0.25 * math.cos(2.0 * math.pi * 31.0 * x))
		}
	}
	mut input := vtl.from_1d[fft.Complex32](values)!
	if strided {
		input = input.slice([0, input_size, 2])!
	}
	for _ in 0 .. 3 {
		_ := fft.fft_f32(input)!
	}
	mut timings := []f64{len: sample_count}
	mut result := fft.fft_f32(input)!
	for sample in 0 .. sample_count {
		start := time.sys_mono_now()
		for _ in 0 .. iterations {
			result = fft.fft_f32(input)!
		}
		elapsed_ns := time.sys_mono_now() - start
		timings[sample] = f64(elapsed_ns) / f64(iterations) / 1000.0
	}
	timings.sort()
	return BenchmarkResult{
		median_us:     timings[sample_count / 2]
		min_us:        timings[0]
		max_us:        timings[sample_count - 1]
		max_abs_error: max_abs_fft_error_f32(size, result)
		checksum:      weighted_fft_checksum_f32(result)
	}
}

fn max_abs_fft_error_f32(size int, result &vtl.Tensor[fft.Complex32]) f64 {
	mut maximum := 0.0
	for i in 0 .. result.size {
		expected_real := 0.0
		expected_imag := if i == 7 {
			-f64(size) / 2.0
		} else if i == size - 7 {
			f64(size) / 2.0
		} else if i == 31 || i == size - 31 {
			f64(size) / 8.0
		} else {
			0.0
		}
		value := result.get_nth(i)
		absolute_error := math.hypot(f64(value.re) - expected_real, f64(value.im) - expected_imag)
		if absolute_error > maximum {
			maximum = absolute_error
		}
	}
	return maximum
}

fn weighted_fft_checksum_f32(result &vtl.Tensor[fft.Complex32]) f64 {
	mut checksum := 0.0
	for i in 0 .. result.size {
		value := result.get_nth(i)
		weight := f64(i + 1)
		checksum += weight * (f64(value.re) + 0.5 * f64(value.im))
	}
	return checksum
}
