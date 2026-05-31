module main

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

// Vulkan smoke: f32 synthetic pipeline + optional linear_forward_vulkan check.
//
//   v -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
//
// Default Sequential forward stays on CPU; Vulkan GEMM is exercised explicitly below.

const batch_size = 4
const epochs = 1
const batches = 2

fn main() {
	println('nn_cifar10_vulkan: build with -d vulkan for GPU linear smoke')

	ctx := autograd.ctx[f32]()
	mut model := models.sequential_from_ctx[f32](ctx)
	model.input([3, 32, 32])
	model.flatten()
	model.linear(10)
	model.softmax()
	model.mse_loss()

	mut opt := optimizers.adam_optimizer[f32](optimizers.AdamOptimizerConfig{
		learning_rate: 0.001
	})
	opt.build_params(model.info.layers)

	for epoch := 0; epoch < epochs; epoch++ {
		for b := 0; b < batches; b++ {
			x_tensor := vtl.ones[f32]([batch_size, 3, 32, 32])
			y_tensor := vtl.zeros[f32]([batch_size, 10])

			x := ctx.variable(x_tensor, requires_grad: true)
			pred := model.forward(x)!
			mut loss := model.loss(pred, y_tensor)!
			loss_val := loss.value.get([0])

			loss.backprop()!
			opt.update()!

			println('epoch ${epoch + 1}/${epochs} batch ${b + 1}/${batches} loss=${loss_val:.4f}')
		}
	}

	vulkan_linear_smoke()
	println('Vulkan CIFAR-shaped pipeline OK ✅')
}

fn vulkan_linear_smoke() {
	x := vtl.ones[f32]([2, 8])
	w := vtl.ones[f32]([4, 8]) * f32(0.1)
	b := vtl.zeros[f32]([4])
	out := layers.linear_forward_vulkan(x, w, b) or {
		println('Vulkan linear skip: ${err}')
		return
	}
	assert out.shape == [2, 4]
	out.release()
	x.release()
	w.release()
	b.release()
	println('Vulkan linear_forward smoke OK')
}
