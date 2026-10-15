module loss

import math
import vtl
import vtl.autograd

fn test_smooth_l1_loss_matches_pytorch_beta_scaling() ! {
	c := autograd.ctx[f64]()
	prediction := c.variable(vtl.from_1d([0.25, 2.0, -0.5])!)
	target := vtl.zeros[f64]([3])
	objective := smooth_l1_loss[f64](beta: 0.5)
	result := objective.loss(prediction, target)!
	// Mean of [0.0625, 1.75, 0.25] is 0.6875.
	assert math.abs(result.value.get_nth(0) - 0.6875) < 1e-12
}

fn test_smooth_l1_loss_backward_scales_inside_beta_and_at_boundary() ! {
	c := autograd.ctx[f64]()
	prediction := c.variable(vtl.from_1d([0.25, 2.0, -0.5])!)
	target := vtl.zeros[f64]([3])
	mut result := smooth_l1_loss[f64](beta: 0.5).loss(prediction, target)!
	result.backprop()!
	assert math.abs(prediction.grad.get_nth(0) - 1.0 / 6.0) < 1e-12
	assert math.abs(prediction.grad.get_nth(1) - 1.0 / 3.0) < 1e-12
	assert math.abs(prediction.grad.get_nth(2) + 1.0 / 3.0) < 1e-12
}

fn test_smooth_l1_loss_supports_f32() ! {
	c := autograd.ctx[f32]()
	prediction := c.variable(vtl.from_1d([f32(0.25), 2.0])!)
	target := vtl.zeros[f32]([2])
	result := smooth_l1_loss[f32](beta: 0.5).loss(prediction, target)!
	assert math.abs(f64(result.value.get_nth(0)) - 0.90625) < 1e-6
}

fn test_smooth_l1_loss_supports_sum_and_none_reductions() ! {
	for reduction in ['sum', 'none'] {
		c := autograd.ctx[f64]()
		prediction := c.variable(vtl.from_array([0.25, -2.0], [2])!)
		target := vtl.zeros[f64]([2])
		mut result := smooth_l1_loss[f64](beta: 0.5, reduction: reduction).loss(prediction, target)!
		if reduction == 'none' {
			assert result.value.shape == [2]
			assert math.abs(result.value.get_nth(0) - 0.0625) < 1e-12
			assert math.abs(result.value.get_nth(1) - 1.75) < 1e-12
		} else {
			assert math.abs(result.value.get_nth(0) - 1.8125) < 1e-12
		}
		result.backprop()!
		assert math.abs(prediction.grad.get_nth(0) - 0.5) < 1e-12
		assert math.abs(prediction.grad.get_nth(1) + 1.0) < 1e-12
	}
}

fn test_smooth_l1_loss_matches_l1_when_beta_is_zero() ! {
	c := autograd.ctx[f64]()
	prediction := c.variable(vtl.from_1d([0.0, 2.0, -0.5])!)
	target := vtl.zeros[f64]([3])
	mut result := smooth_l1_loss[f64](beta: 0.0).loss(prediction, target)!
	assert math.abs(result.value.get_nth(0) - 2.5 / 3.0) < 1e-12
	result.backprop()!
	assert prediction.grad.get_nth(0) == 0.0
	assert math.abs(prediction.grad.get_nth(1) - 1.0 / 3.0) < 1e-12
	assert math.abs(prediction.grad.get_nth(2) + 1.0 / 3.0) < 1e-12
}

fn test_smooth_l1_loss_empty_reductions_match_pytorch() ! {
	c := autograd.ctx[f64]()
	target := vtl.zeros[f64]([0])
	mut mean_result := smooth_l1_loss[f64]().loss(c.variable(vtl.zeros[f64]([0])), target)!
	assert math.is_nan(mean_result.value.get_nth(0))
	mut sum_result := smooth_l1_loss[f64](reduction: 'sum').loss(c.variable(vtl.zeros[f64]([0])), target)!
	assert sum_result.value.get_nth(0) == 0.0
	none_result := smooth_l1_loss[f64](reduction: 'none').loss(c.variable(vtl.zeros[f64]([0])), target)!
	assert none_result.value.shape == [0]
}

fn test_smooth_l1_loss_rejects_invalid_beta_and_mismatched_inputs() ! {
	c := autograd.ctx[f64]()
	prediction := c.variable(vtl.from_1d([1.0, 2.0])!)
	target := vtl.zeros[f64]([2])
	for beta in [-1.0, math.inf(1), math.nan()] {
		smooth_l1_loss[f64](beta: beta).loss(prediction, target) or {
			assert err.msg().contains('beta must be finite and non-negative')
			continue
		}
		assert false, 'invalid beta ${beta} should fail'
	}
	smooth_l1_loss[f64](reduction: 'invalid').loss(prediction, target) or {
		assert err.msg().contains('reduction must be none, mean, or sum')
		return
	}
	assert false, 'invalid reduction should fail'

	smooth_l1_loss[f64]().loss(prediction, vtl.zeros[f64]([1])) or {
		assert err.msg().contains('identical shapes')
		return
	}
	assert false, 'mismatched shapes should fail'
}
