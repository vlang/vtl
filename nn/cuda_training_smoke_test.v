module nn

import os
import vtl
import vtl.autograd
import vtl.autograd_cuda
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

// Placeholder for `v test` without `-d cuda` (CI scoped nn tests).
fn test_cuda_training_smoke_needs_cuda_build() {
	assert true
}

$if cuda ? {
	// f64 training smoke with DeviceSession (`VTL_TEST_CUDA=1`, `-d cuda`).
	fn test_cuda_training_smoke_two_batches() ! {
		if !layers.cuda_tests_enabled() {
			return
		}
		mut ctx := autograd.ctx[f64]()
		autograd_cuda.attach_context_session(mut ctx)
		mut model := models.sequential_from_ctx[f64](ctx)
		model.input([3, 4, 4])
		model.conv2d(3, 4, [2, 2], layers.Conv2DConfig{padding: [0, 0]})
		model.flatten()
		model.linear(3)
		model.mse_loss()

		mut opt := optimizers.adam_optimizer[f64](optimizers.AdamOptimizerConfig{
			learning_rate: 0.001
		})
		opt.build_params(model.info.layers)

		for b in 0 .. 2 {
			x := ctx.variable(vtl.ones[f64]([1, 3, 4, 4]), requires_grad: true)
			y := vtl.zeros[f64]([1, 3])
			pred := model.forward(x)!
			mut loss := model.loss(pred, y)!
			loss.backprop()!
			opt.update()!
			_ = loss.value.get([0])
			_ = b
		}
	}
}
