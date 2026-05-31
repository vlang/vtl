// VTL Conv2D forward benchmark (CPU f64 path).
// Run: v run vtl/benchmarks/vs_numpy/conv2d_bench.v
module main

import time
import vtl
import vtl.nn.internal
import vtl.benchmarks.util as bu

fn main() {
	bu.print_header('VTL Conv2D forward benchmark (CPU f64)')
	config := bu.BenchConfig{
		iterations:  5
		warmup_runs: 2
	}
	bu.print_table_header()
	bench_conv2d(config)
}

fn bench_conv2d(config bu.BenchConfig) {
	// [1, 1, 32, 32] input, 3x3 kernel, same padding
	input := vtl.random[f64](0.0, 1.0, [1, 1, 32, 32], vtl.TensorData{})
	weight := vtl.random[f64](0.0, 1.0, [1, 1, 3, 3], vtl.TensorData{})
	bias := vtl.zeros[f64]([1, 1])
	cfg := internal.Conv2DConfig{
		padding:  [1, 1]
		stride:   [1, 1]
		dilation: [1, 1]
		groups:   1
	}
	k := [3, 3]

	for _ in 0 .. config.warmup_runs {
		internal.conv2d_forward_f64(input, weight, bias, k, cfg) or { panic(err) }
	}

	mut samples := []f64{len: config.iterations}
	for i in 0 .. config.iterations {
		t0 := time.ticks()
		internal.conv2d_forward_f64(input, weight, bias, k, cfg) or { panic(err) }
		samples[i] = f64(time.ticks() - t0)
	}
	avg := bu.mean_time_ms(mut samples)
	bu.print_row('conv2d', '1x1x32x32', avg, '—')
}
