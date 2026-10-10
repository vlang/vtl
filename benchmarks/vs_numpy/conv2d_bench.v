// VTL Conv2D forward benchmark (CPU f64 path).
// Compile with `-prod` from ~/.vmodules, then run the resulting executable.
module main

import time
import vtl
import vtl.nn.internal
import vtl.benchmarks.util as bu

const benchmark_batch_size = 1
const benchmark_in_channels = 4
const benchmark_out_channels = 8
const benchmark_image_height = 32
const benchmark_image_width = 32
const benchmark_kernel_height = 3
const benchmark_kernel_width = 3

fn main() {
	bu.print_header('VTL Conv2D forward benchmark (CPU f64; compare with NumPy)')
	config := bu.BenchConfig{
		iterations:  20
		warmup_runs: 3
	}
	bu.print_table_header()
	bench_conv2d(config) or { panic(err) }
}

fn bench_conv2d(config bu.BenchConfig) ! {
	// Deterministic NCHW input and OIHW weights also used by the NumPy baseline.
	mut input_values := []f64{len: benchmark_batch_size * benchmark_in_channels * benchmark_image_height * benchmark_image_width}
	for i in 0 .. input_values.len {
		input_values[i] = f64((i * 13) % 101 - 50) / 101.0
	}
	mut weight_values := []f64{len: benchmark_out_channels * benchmark_in_channels * benchmark_kernel_height * benchmark_kernel_width}
	for i in 0 .. weight_values.len {
		weight_values[i] = f64((i * 7) % 37 - 18) / 37.0
	}
	mut bias_values := []f64{len: benchmark_out_channels}
	for i in 0 .. bias_values.len {
		bias_values[i] = f64(i + 1) / 10.0
	}
	input := vtl.from_array(input_values, [benchmark_batch_size, benchmark_in_channels,
		benchmark_image_height, benchmark_image_width])!
	weight := vtl.from_array(weight_values, [benchmark_out_channels, benchmark_in_channels,
		benchmark_kernel_height, benchmark_kernel_width])!
	bias := vtl.from_array(bias_values, [1, benchmark_out_channels])!
	cfg := internal.Conv2DConfig{
		padding:  [1, 1]
		stride:   [1, 1]
		dilation: [1, 1]
		groups:   1
	}
	k := [benchmark_kernel_height, benchmark_kernel_width]

	for _ in 0 .. config.warmup_runs {
		_ = internal.conv2d_forward_f64(input, weight, bias, k, cfg) or { panic(err) }
	}

	mut samples := []f64{len: config.iterations}
	mut output := internal.conv2d_forward_f64(input, weight, bias, k, cfg) or { panic(err) }
	for i in 0 .. config.iterations {
		started := time.sys_mono_now()
		output = internal.conv2d_forward_f64(input, weight, bias, k, cfg) or { panic(err) }
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
	}
	avg := bu.mean_time_ms(mut samples)
	mut checksum := 0.0
	for value in output.to_array() {
		checksum += value
	}
	bu.print_row('conv2d', '1x4x32x32, 8x4x3x3', avg, '-')
	println('Checksum: ${checksum:.12f}')
	println('NumPy reference: benchmarks/vs_numpy/numpy_conv2d_baseline.py')
}
