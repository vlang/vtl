module autograd_cuda

import os
import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

// f64 Sequential training smoke with DeviceSession (run with VTL_TEST_CUDA=1 -d cuda).
fn test_cuda_training_smoke_two_batches() ! {
	if os.getenv('VTL_TEST_CUDA') != '1' {
		return
	}
	mut ctx := autograd.ctx[f64]()
	attach_context_session(mut ctx)
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([3, 8, 8])
	model.conv2d(3, 4, [3, 3], layers.Conv2DConfig{
		padding: [1, 1]
	})
	model.relu()
	model.flatten()
	model.linear(3)
	model.mse_loss()

	mut opt := optimizers.adam_optimizer[f64](optimizers.AdamOptimizerConfig{
		learning_rate: 0.001
	})
	opt.build_params(model.info.layers)

	for b in 0 .. 2 {
		x := ctx.variable(vtl.ones[f64]([1, 3, 8, 8]), requires_grad: true)
		y := vtl.zeros[f64]([1, 3])
		pred := model.forward(x)!
		mut loss := model.loss(pred, y)!
		loss.backprop()!
		opt.update()!
		_ = loss.value.get([0])
		_ = b
	}
}
