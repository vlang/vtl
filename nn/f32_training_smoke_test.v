module nn

import vtl
import vtl.autograd
import vtl.nn.models
import vtl.nn.optimizers

// f32 end-to-end training step: forward, MSE, backprop, Adam update.
fn test_sequential_f32_mse_train_adam_one_step() ! {
	ctx := autograd.ctx[f32]()
	mut model := models.sequential_from_ctx[f32](ctx)
	model.input([4])
	model.linear(3)
	model.mse_loss()

	mut opt := optimizers.adam_optimizer[f32](optimizers.AdamOptimizerConfig{
		learning_rate: 0.1
	})
	opt.build_params(model.info.layers)
	assert opt.params.len == 2, 'expected linear weight and bias to be trainable'
	initial_params := opt.params.map(it.value.to_array())

	x := ctx.variable(vtl.ones[f32]([1, 4]))
	y := vtl.zeros[f32]([1, 3])
	pred := model.forward(x)!
	mut loss := model.loss(pred, y)!
	assert loss.value.get_nth(0) >= 0.0, 'MSE loss must be non-negative'
	loss.backprop()!
	opt.update()!
	mut parameter_changed := false
	for i, parameter in opt.params {
		if parameter.value.to_array() != initial_params[i] {
			parameter_changed = true
		}
		for gradient in parameter.grad.to_array() {
			assert gradient == 0.0, 'Adam must clear gradients after the update'
		}
	}
	assert parameter_changed, 'Adam update must change at least one f32 model parameter'
}
