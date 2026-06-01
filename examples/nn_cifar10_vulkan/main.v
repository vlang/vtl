module main

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

// CIFAR-shaped f32 training smoke: flatten + Linear (Vulkan GEMM when opted in).
//
//   v run vtl/examples/nn_cifar10_vulkan/main.v
//   VTL_USE_VULKAN=1 v -prod -d vulkan run vtl/examples/nn_cifar10_vulkan/main.v
//
// Linear backward uses Vulkan GEMM when VTL_USE_VULKAN=1; Adam stays on CPU.

const batch_size = 2
const epochs = 1
const batches = 2

fn main() {
	use_vk := layers.vulkan_linear_enabled()
	println('nn_cifar10_vulkan: VTL_USE_VULKAN=${use_vk} (build with -d vulkan; use -prod for GPU)')

	ctx := autograd.ctx[f32]()
	mut model := models.sequential_from_ctx[f32](ctx)
	model.input([3, 8, 8])
	model.conv2d(3, 8, [3, 3], layers.Conv2DConfig{padding: [1, 1], stride: [1, 1]})
	model.flatten()
	model.linear(10)
	model.mse_loss()

	mut opt := optimizers.adam_optimizer[f32](optimizers.AdamOptimizerConfig{
		learning_rate: 0.01
	})
	opt.build_params(model.info.layers)

	for epoch := 0; epoch < epochs; epoch++ {
		for b := 0; b < batches; b++ {
			x_tensor := vtl.ones[f32]([batch_size, 3, 8, 8])
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

	println('f32 Vulkan CIFAR-shaped training smoke OK ✅')
}
