module optimizers

import vtl
import vtl.autograd

fn test_adagrad_checkpoint_restore_preserves_optimizer_state() ! {
	left_context := autograd.ctx[f64]()
	right_context := autograd.ctx[f64]()
	left_parameter := left_context.variable(vtl.from_1d([5.0])!)
	right_parameter := right_context.variable(vtl.from_1d([5.0])!)
	mut left := adagrad[f64](learning_rate: 0.1, weight_decay: 0.02)
	left.params << left_parameter
	left.accumulated_sq_grads << vtl.zeros_like[f64](left_parameter.grad)
	left_parameter.grad.set_nth(0, 1.0)
	left.update()!
	config := left.checkpoint_config()
	mut accumulators := []&vtl.Tensor[f64]{}
	for accumulator in left.accumulated_sq_grads {
		accumulators << accumulator.copy(.row_major)
	}
	right_parameter.value.set_nth(0, left_parameter.value.get_nth(0))

	mut right := adagrad[f64](learning_rate: 0.5)
	right.params << right_parameter
	right.restore_checkpoint_state(config, accumulators)!
	assert right.checkpoint_config() == config
	assert right.accumulated_sq_grads[0].to_array() == left.accumulated_sq_grads[0].to_array()

	left_parameter.grad.set_nth(0, 0.5)
	right_parameter.grad.set_nth(0, 0.5)
	left.update()!
	right.update()!
	assert left_parameter.value.to_array() == right_parameter.value.to_array()
	assert left.accumulated_sq_grads[0].to_array() == right.accumulated_sq_grads[0].to_array()
}
