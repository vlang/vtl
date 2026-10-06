// VTL complex f32 FFT benchmark. Run from ~/.vmodules with:
// v -prod run ./vtl/benchmarks/vs_numpy/fft_complex_f32_bench.v
module main

import math
import time
import vtl
import vtl.fft

const sizes = [256, 1024, 4096, 16384]

fn main() {
	println('VTL complex f32 FFT forward transform')
	println('size,iterations,mean_us,checksum')
	for size in sizes {
		iterations := if size <= 1024 {
			50
		} else if size <= 4096 {
			20
		} else {
			10
		}
		mean_us, checksum := benchmark_fft_f32(size, iterations) or {
			eprintln('complex f32 FFT benchmark failed for n=${size}: ${err}')
			exit(1)
		}
		println('${size},${iterations},${mean_us:.3f},${checksum:.6f}')
	}
}

fn benchmark_fft_f32(size int, iterations int) !(f64, f64) {
	mut values := []fft.Complex32{len: size}
	for i in 0 .. size {
		x := f64(i) / f64(size)
		values[i] = fft.Complex32{
			re: f32(math.sin(2.0 * math.pi * 7.0 * x))
			im: f32(0.25 * math.cos(2.0 * math.pi * 31.0 * x))
		}
	}
	input := vtl.from_1d[fft.Complex32](values)!
	for _ in 0 .. 3 {
		_ := fft.fft_f32(input)!
	}
	mut checksum := 0.0
	start := time.sys_mono_now()
	for _ in 0 .. iterations {
		result := fft.fft_f32(input)!
		checksum += f64(result.get_nth(0).re)
	}
	elapsed_ns := time.sys_mono_now() - start
	mean_us := f64(elapsed_ns) / f64(iterations) / 1000.0
	return mean_us, checksum
}
