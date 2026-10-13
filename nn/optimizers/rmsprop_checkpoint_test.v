module optimizers

import vtl
import vtl.autograd

fn test_rmsprop_checkpoint_restore_preserves_optimizer_state() ! {
	left_context := autograd.ctx[f64]()
	right_context := autograd.ctx[f64]()
	left_parameter := left_context.variable(vtl.from_1d([5.0])!)
	right_parameter := right_context.variable(vtl.from_1d([5.0])!)
	mut left := rmsprop[f64](learning_rate: 0.1, alpha: 0.8, weight_decay: 0.02)
	left.params << left_parameter
	left.sq_avg << vtl.zeros_like[f64](left_parameter.grad)
	left_parameter.grad.set_nth(0, 1.0)
	left.update()!
	config := left.checkpoint_config()
	mut sq_avg := []&vtl.Tensor[f64]{}
	for buffer in left.sq_avg {
		sq_avg << buffer.copy(.row_major)
	}
	right_parameter.value.set_nth(0, left_parameter.value.get_nth(0))

	mut right := rmsprop[f64](learning_rate: 0.5)
	right.params << right_parameter
	right.restore_checkpoint_state(config, sq_avg)!
	assert right.checkpoint_config() == config
	assert right.sq_avg[0].to_array() == left.sq_avg[0].to_array()

	left_parameter.grad.set_nth(0, 0.5)
	right_parameter.grad.set_nth(0, 0.5)
	left.update()!
	right.update()!
	assert left_parameter.value.to_array() == right_parameter.value.to_array()
	assert left.sq_avg[0].to_array() == right.sq_avg[0].to_array()
}
