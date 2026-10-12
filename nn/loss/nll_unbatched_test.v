module loss

import math
import vtl
import vtl.autograd

fn test_nll_loss_supports_unbatched_scalar_targets_and_reductions() ! {
	for reduction in ['none', 'mean', 'sum'] {
		ctx := autograd.ctx[f64]()
		prediction := ctx.variable(vtl.from_array([-2.0, -0.5, -1.0], [3])!)
		target := vtl.from_array([1], [])!
		weights := vtl.from_array([2.0, 4.0, 8.0], [3])!
		mut result := nll_loss_indices[f64](weights, reduction: reduction).loss(prediction, target)!
		assert result.value.shape == [], 'unbatched NLL reduction ${reduction} must return a scalar'
		want_loss := if reduction == 'mean' { 0.5 } else { 2.0 }
		assert math.abs(result.value.get_nth(0) - want_loss) < 1e-12
		result.backprop()!
		want_gradient := if reduction == 'mean' { -1.0 } else { -4.0 }
		assert prediction.grad.get_nth(0) == 0.0
		assert math.abs(prediction.grad.get_nth(1) - want_gradient) < 1e-12
		assert prediction.grad.get_nth(2) == 0.0
	}
}

fn test_nll_loss_unbatched_generic_scalar_target_and_ignore_index() ! {
	ctx := autograd.ctx[f64]()
	prediction := ctx.variable(vtl.from_array([-2.0, -0.5, -1.0], [3])!)
	target := vtl.from_array([-100.0], [])!
	mut result := nll_loss_with_options[f64](unsafe { nil }, ignore_index: -100, reduction: 'none').loss(prediction,
		target)!
	assert result.value.shape == []
	assert result.value.get_nth(0) == 0.0
	result.backprop()!
	for i in 0 .. prediction.grad.size() {
		assert prediction.grad.get_nth(i) == 0.0
	}

	ctx2 := autograd.ctx[f64]()
	prediction2 := ctx2.variable(vtl.from_array([-2.0, -0.5, -1.0], [3])!)
	soft_target := vtl.from_array([0.0, 1.0, 0.0], [3])!
	mut soft_result := nll_loss[f64](unsafe { nil }).loss(prediction2, soft_target)!
	assert soft_result.value.shape == []
	assert math.abs(soft_result.value.get_nth(0) - 0.5) < 1e-12
	soft_result.backprop()!
	assert prediction2.grad.get_nth(0) == 0.0
	assert math.abs(prediction2.grad.get_nth(1) + 1.0) < 1e-12
	assert prediction2.grad.get_nth(2) == 0.0
}
