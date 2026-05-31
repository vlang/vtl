// VTL 3-layer MLP backprop benchmark (f64, CPU autograd).
// Run: v run vtl/benchmarks/vs_numpy/autograd_bench.v
module main

import time
import vtl
import vtl.autograd
import vtl.nn.models
import vtl.benchmarks.util as bu

const batch_size = 64
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
	ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([input_dim])
	model.linear(hidden_dim)
	model.relu()
	model.linear(hidden_dim)
	model.relu()
	model.linear(output_dim)
	model.mse_loss()

	x_data := vtl.ones[f64]([batch, input_dim])
	y_data := vtl.zeros[f64]([batch, output_dim])
	mut x := ctx.variable(x_data, requires_grad: true)

	for _ in 0 .. config.warmup_runs {
		run_step(mut model, x, y_data)!
	}

	mut samples := []f64{len: config.iterations}
	for i in 0 .. config.iterations {
		t0 := time.ticks()
		run_step(mut model, x, y_data)!
		samples[i] = f64(time.ticks() - t0)
	}
	avg := bu.mean_time_ms(mut samples)
	bu.print_row('mlp_backprop', '${batch}x${input_dim}', avg, '-')
}

fn run_step[T](mut model models.Sequential[T], x &autograd.Variable[T], y &vtl.Tensor[T]) ! {
	pred := model.forward(x)!
	mut loss := model.loss(pred, y)!
	loss.backprop()!
}
