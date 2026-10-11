// VTL 3-layer MLP backprop benchmark (f64, CPU autograd).
// Compile with `-prod` from ~/.vmodules, then run the resulting executable.
module main

import time
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.loss
import vtl.nn.types
import vtl.benchmarks.util as bu

const input_dim = 128
const hidden_dim = 64
const output_dim = 32
const sizes = [32, 64]

fn main() {
	bu.print_header('VTL autograd MLP backprop (3-layer, f64)')
	config := bu.BenchConfig{
		iterations:  10
		warmup_runs: 5
	}
	bu.print_table_header()
	for n in sizes {
		bench_mlp_backprop(n, config)!
	}
	println('\nCompare with PyTorch: uv run --with torch python ./vtl/benchmarks/vs_numpy/pytorch_baseline.py autograd')
}

fn bench_mlp_backprop(batch int, config bu.BenchConfig) ! {
	vtl.random_seed(42)
	ctx := autograd.ctx[f64]()
	objective := loss.mse_loss[f64]()
	first := linear_layer(ctx, input_dim, hidden_dim)
	second := linear_layer(ctx, hidden_dim, hidden_dim)
	third := linear_layer(ctx, hidden_dim, output_dim)
	relu1 := layers.relu_layer[f64](ctx, [hidden_dim])
	relu2 := layers.relu_layer[f64](ctx, [hidden_dim])

	x_data := vtl.ones[f64]([batch, input_dim])
	y_data := vtl.zeros[f64]([batch, output_dim])
	mut x := ctx.variable(x_data)

	for _ in 0 .. config.warmup_runs {
		run_step(first, second, third, relu1, relu2, objective, x, y_data)!
	}

	mut samples := []f64{len: config.iterations}
	for i in 0 .. config.iterations {
		t0 := time.sys_mono_now()
		run_step(first, second, third, relu1, relu2, objective, x, y_data)!
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

fn run_step(first &layers.LinearLayer[f64], second &layers.LinearLayer[f64], third &layers.LinearLayer[f64], relu1 types.Layer[f64], relu2 types.Layer[f64], objective types.Loss[f64], x &autograd.Variable[f64], target &vtl.Tensor[f64]) ! {
	mut first_weights := first.weights
	mut first_bias := first.bias
	mut second_weights := second.weights
	mut second_bias := second.bias
	mut third_weights := third.weights
	mut third_bias := third.bias
	zero_grad(mut first_weights)
	zero_grad(mut first_bias)
	zero_grad(mut second_weights)
	zero_grad(mut second_bias)
	zero_grad(mut third_weights)
	zero_grad(mut third_bias)
	hidden1 := relu1.forward(first.forward(x)!)!
	hidden2 := relu2.forward(second.forward(hidden1)!)!
	prediction := third.forward(hidden2)!
	mut loss_result := objective.loss(prediction, target)!
	loss_result.backprop()!
}

fn zero_grad(mut variable &autograd.Variable[f64]) {
	mut gradient := variable.grad
	gradient.fill(0.0)
}
