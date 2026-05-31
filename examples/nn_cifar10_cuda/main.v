module main

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

// GPU smoke: tiny synthetic CIFAR + Conv2D + Linear on CUDA when opted in.
//
//   VTL_USE_CUDA=1 v -d cuda run vtl/examples/nn_cifar10_cuda/main.v
//
// CI does not run this example (requires CUDA toolchain + GPU libs).

const batch_size = 4
const epochs = 1
const batches = 2

fn main() {
	use_cuda := layers.cuda_linear_enabled()
	println('nn_cifar10_cuda: VTL_USE_CUDA=${use_cuda} (-d cuda build for GPU forward)')

	ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([3, 32, 32])
	model.conv2d(3, 8, [3, 3], layers.Conv2DConfig{
		padding: [1, 1]
	})
	model.relu()
	model.flatten()
	model.linear(10)
	model.softmax()
	model.mse_loss()

	mut opt := optimizers.adam_optimizer[f64](optimizers.AdamOptimizerConfig{
		learning_rate: 0.001
	})
	opt.build_params(model.info.layers)

	for epoch := 0; epoch < epochs; epoch++ {
		for b := 0; b < batches; b++ {
			x_tensor := vtl.ones[f64]([batch_size, 3, 32, 32])
			y_tensor := vtl.zeros[f64]([batch_size, 10])

			x := ctx.variable(x_tensor, requires_grad: true)
			pred := model.forward(x)!
			mut loss := model.loss(pred, y_tensor)!
			loss_val := loss.value.get([0])

			loss.backprop()!
			opt.update()!

			println('epoch ${epoch + 1}/${epochs} batch ${b + 1}/${batches} loss=${loss_val:.4f}')
		}
	}

	if use_cuda {
		println('CUDA opt-in training smoke OK ✅')
	} else {
		println('CPU path OK (set VTL_USE_CUDA=1 -d cuda for GPU) ✅')
	}
}
