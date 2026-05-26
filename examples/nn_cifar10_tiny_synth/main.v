module main

import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

const batch_size = 4
const epochs = 1
const batches = 2

fn main() {
	ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([3, 32, 32])
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
			// Synthetic CIFAR-like batch: [B, C, H, W]
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

	println('Tiny synthetic CIFAR pipeline OK ✅')
}
