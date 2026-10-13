module optimizers

import vtl
import vtl.autograd

fn test_radam_checkpoint_restore_preserves_optimizer_state() ! {
	left_context := autograd.ctx[f64]()
	right_context := autograd.ctx[f64]()
	left_parameter := left_context.variable(vtl.from_1d([5.0])!)
	right_parameter := right_context.variable(vtl.from_1d([5.0])!)
	mut left := radam_optimizer[f64](
		learning_rate:          0.1
		weight_decay:           0.02
		decoupled_weight_decay: true
	)
	left.params << left_parameter
	left.first_moments << vtl.zeros_like[f64](left_parameter.grad)
	left.second_moments << vtl.zeros_like[f64](left_parameter.grad)
	left_parameter.grad.set_nth(0, 1.0)
	left.update()!
	config := left.checkpoint_config()
	mut first_moments := []&vtl.Tensor[f64]{}
	mut second_moments := []&vtl.Tensor[f64]{}
	for moment in left.first_moments {
		first_moments << moment.copy(.row_major)
	}
	for moment in left.second_moments {
		second_moments << moment.copy(.row_major)
	}
	right_parameter.value.set_nth(0, left_parameter.value.get_nth(0))

	mut right := radam_optimizer[f64](learning_rate: 0.5)
	right.params << right_parameter
	right.restore_checkpoint_state(config, first_moments, second_moments)!
	assert right.checkpoint_config() == config
	left_parameter.grad.set_nth(0, 0.5)
	right_parameter.grad.set_nth(0, 0.5)
	left.update()!
	right.update()!
	assert left_parameter.value.to_array() == right_parameter.value.to_array()
	assert left.step == right.step
	assert left.first_moments[0].to_array() == right.first_moments[0].to_array()
	assert left.second_moments[0].to_array() == right.second_moments[0].to_array()
}
