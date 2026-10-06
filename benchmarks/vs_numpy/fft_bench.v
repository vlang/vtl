// VTL real FFT benchmark. Run from ~/.vmodules with:
// v -prod run ./vtl/benchmarks/vs_numpy/fft_bench.v
module main

import math
import time
import vtl
import vtl.fft

const sizes = [256, 1024, 4096, 16384]

fn main() {
	println('VTL real f64 rfft forward transform')
	println('size,iterations,mean_us,checksum')
	for size in sizes {
		iterations := iterations_for(size)
		mean_us, checksum := benchmark_rfft(size, iterations) or {
			eprintln('rfft benchmark failed for n=${size}: ${err}')
			exit(1)
		}
		println('${size},${iterations},${mean_us:.3f},${checksum:.6f}')
	}
	println('VTL real f32-input rfft with complex f32 output')
	println('size,iterations,mean_us,checksum')
	for size in sizes {
		iterations := iterations_for(size)
		mean_us, checksum := benchmark_rfft_f32(size, iterations) or {
			eprintln('f32 rfft benchmark failed for n=${size}: ${err}')
			exit(1)
		}
		println('${size},${iterations},${mean_us:.3f},${checksum:.6f}')
	}
}

fn iterations_for(size int) int {
	return match size {
		256 { 100 }
		1024 { 50 }
		4096 { 20 }
		else { 10 }
	}
}

fn benchmark_rfft(size int, iterations int) !(f64, f64) {
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
	mut checksum := 0.0
	start := time.sys_mono_now()
	for _ in 0 .. iterations {
		result := plan.forward(input)!
		checksum += result.get_nth(0).re
	}
	elapsed_ns := time.sys_mono_now() - start
	mean_us := f64(elapsed_ns) / f64(iterations) / 1000.0
	return mean_us, checksum
}

fn benchmark_rfft_f32(size int, iterations int) !(f64, f64) {
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
	mut checksum := 0.0
	start := time.sys_mono_now()
	for _ in 0 .. iterations {
		result := plan.forward(input)!
		checksum += f64(result.get_nth(0).re)
	}
	elapsed_ns := time.sys_mono_now() - start
	mean_us := f64(elapsed_ns) / f64(iterations) / 1000.0
	return mean_us, checksum
}
