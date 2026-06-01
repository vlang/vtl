module main

import os
import vtl
import vtl.autograd
import vtl.autograd_cuda
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

// GPU smoke: f64 Sequential (flatten + Linear), backprop + Adam on DeviceSession.
// Conv2D CUDA is covered in device_session / nn tests; full conv+attach backprop is heavy on CPU.
//
//   VTL_USE_CUDA=1 v -d cuda run vtl/examples/nn_cifar10_cuda/main.v
//
// Optional Phase 2 chain: export VTL_GPU_ACTIVATIONS=1
// CI does not run this example (CUDA link + memory).

const batch_size = 2
const epochs = 1
const batches = 2

fn main() {
	use_cuda := layers.cuda_linear_enabled()
	gpu_act := os.getenv('VTL_GPU_ACTIVATIONS') == '1'
	println('nn_cifar10_cuda: VTL_USE_CUDA=${use_cuda} VTL_GPU_ACTIVATIONS=${gpu_act}')
	println('Build with -d cuda for GPU forward paths')

	mut ctx := autograd.ctx[f64]()
	autograd_cuda.attach_context_session(mut ctx)

	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([3, 8, 8])
	model.flatten()
	model.linear(10)
	model.mse_loss()

	mut opt := optimizers.adam_optimizer[f64](optimizers.AdamOptimizerConfig{
		learning_rate: 0.001
	})
	opt.build_params(model.info.layers)

	for epoch := 0; epoch < epochs; epoch++ {
		for b := 0; b < batches; b++ {
			x_tensor := vtl.ones[f64]([batch_size, 3, 8, 8])
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
		println('CUDA DeviceSession training smoke OK ✅')
	} else {
		println('CPU path OK (set VTL_USE_CUDA=1 and -d cuda for GPU) ✅')
	}
}
