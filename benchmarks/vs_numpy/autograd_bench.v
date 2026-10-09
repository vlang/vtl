// VTL 3-layer MLP backprop benchmark (f64, CPU autograd).
// Run: v run vtl/benchmarks/vs_numpy/autograd_bench.v
module main

import time
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.benchmarks.util as bu

const input_dim = 128
const hidden_dim = 64
const output_dim = 32
const sizes = [32, 64]

fn main() {
	bu.print_header('VTL autograd MLP backprop (3-layer, f64)')
	config := bu.BenchConfig{
		iterations:  2
		warmup_runs: 1
	}
	bu.print_table_header()
	for n in sizes {
		bench_mlp_backprop(n, config)!
	}
	println('\nCompare with PyTorch: python3 vtl/benchmarks/vs_numpy/pytorch_baseline.py autograd')
}

fn bench_mlp_backprop(batch int, config bu.BenchConfig) ! {
	vtl.random_seed(42)
	ctx := autograd.ctx[f64]()
	first := linear_layer(ctx, input_dim, hidden_dim)
	second := linear_layer(ctx, hidden_dim, hidden_dim)
	third := linear_layer(ctx, hidden_dim, output_dim)

	x_data := vtl.ones[f64]([batch, input_dim])
	y_data := vtl.zeros[f64]([batch, output_dim])
	mut x := ctx.variable(x_data)
	y := ctx.variable(y_data, requires_grad: false)

	for _ in 0 .. config.warmup_runs {
		run_step(first, second, third, x, y)!
	}

	mut samples := []f64{len: config.iterations}
	for i in 0 .. config.iterations {
		t0 := time.sys_mono_now()
		run_step(first, second, third, x, y)!
		samples[i] = f64(time.sys_mono_now() - t0) / 1_000_000.0
	}
	avg := bu.mean_time_ms(mut samples)
	bu.print_row('mlp_backprop', '${batch}x${input_dim}', avg, '-')
}

fn linear_layer(ctx &autograd.Context[f64], input_size int, output_size int) &layers.LinearLayer[f64] {
	weights := ctx.variable(vtl.random[f64](-0.05, 0.05, [output_size, input_size], vtl.TensorData{}))
	bias := ctx.variable(vtl.zeros[f64]([1, output_size]))
	return &layers.LinearLayer[f64]{
		weights: weights
		bias:    bias
	}
}

fn run_step(first &layers.LinearLayer[f64], second &layers.LinearLayer[f64], third &layers.LinearLayer[f64], x &autograd.Variable[f64], y &autograd.Variable[f64]) ! {
	hidden1 := first.forward(x)!.clamp(0.0, 1e100)!
	hidden2 := second.forward(hidden1)!.clamp(0.0, 1e100)!
	prediction := third.forward(hidden2)!
	residual := prediction.subtract(y)!
	squared_error := residual.multiply(residual)!
	mut loss := squared_error.mean()!
	loss.backprop()!
}
