module optimizers

import vtl
import vtl.autograd

fn test_sgd_checkpoint_restore_preserves_first_step_momentum_state() ! {
	left_context := autograd.ctx[f64]()
	right_context := autograd.ctx[f64]()
	left_parameter := left_context.variable(vtl.from_1d([5.0])!)
	right_parameter := right_context.variable(vtl.from_1d([5.0])!)
	mut left := sgd[f64](learning_rate: 0.1, momentum: 0.9, dampening: 0.5)
	left.params << left_parameter
	left.momentum_buffers << vtl.zeros_like[f64](left_parameter.grad)
	left.momentum_initialized << false
	config := left.checkpoint_config()
	buffers := left.momentum_buffers.clone()

	mut right := sgd[f64](learning_rate: 0.5)
	right.params << right_parameter
	right.restore_checkpoint_state(config, buffers)!
	assert right.momentum_initialized == [false]
	assert right.momentum_buffers[0].get_nth(0) == 0.0

	left_parameter.grad.set_nth(0, 1.0)
	right_parameter.grad.set_nth(0, 1.0)
	left.update()!
	right.update()!
	assert left_parameter.value.to_array() == right_parameter.value.to_array()
	assert left.momentum_buffers[0].to_array() == right.momentum_buffers[0].to_array()
	assert right.momentum_initialized == [true]
}
