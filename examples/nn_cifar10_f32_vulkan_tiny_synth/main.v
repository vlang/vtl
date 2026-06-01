module main

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

// f32 training smoke; Linear forward uses Vulkan GEMM when opted in.
//
//   VTL_USE_VULKAN=1 v -d vulkan run vtl/examples/nn_cifar10_f32_vulkan_tiny_synth/main.v
//
// Linear backward uses Vulkan GEMM when VTL_USE_VULKAN=1 (-d vulkan); Adam on CPU.

const batch_size = 2
const epochs = 1
const batches = 2

fn main() {
	use_vk := layers.vulkan_linear_enabled()
	println('f32_vulkan_tiny: VTL_USE_VULKAN=${use_vk} (build with -d vulkan for GPU GEMM)')

	ctx := autograd.ctx[f32]()
	mut model := models.sequential_from_ctx[f32](ctx)
	model.input([3, 8, 8])
	model.flatten()
	model.linear(4)
	model.mse_loss()

	mut opt := optimizers.adam_optimizer[f32](optimizers.AdamOptimizerConfig{
		learning_rate: 0.01
	})
	opt.build_params(model.info.layers)

	for epoch := 0; epoch < epochs; epoch++ {
		for b := 0; b < batches; b++ {
			x_tensor := vtl.ones[f32]([batch_size, 3, 8, 8])
			y_tensor := vtl.zeros[f32]([batch_size, 4])

			x := ctx.variable(x_tensor, requires_grad: true)
			pred := model.forward(x)!
			mut loss := model.loss(pred, y_tensor)!
			loss_val := loss.value.get([0])

			loss.backprop()!
			opt.update()!

			println('epoch ${epoch + 1}/${epochs} batch ${b + 1}/${batches} loss=${loss_val:.4f}')
		}
	}

	println('f32 Vulkan tiny synthetic training OK ✅')
}
