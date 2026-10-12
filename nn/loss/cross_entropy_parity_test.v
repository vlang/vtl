module loss

import math
import vtl
import vtl.autograd

fn test_cross_entropy_unbatched_integer_target_none_reduction_and_gradient() ! {
	ctx := autograd.ctx[f64]()
	prediction := ctx.variable(vtl.from_array([0.0, math.log(3.0)], [2])!)
	target := vtl.from_array([1], [])!
	mut result := cross_entropy_loss_indices[f64](unsafe { nil }, reduction: 'none').loss(prediction,
		target)!
	assert result.value.shape == []
	assert math.abs(result.value.get_nth(0) - math.log(4.0 / 3.0)) < 1e-12
	result.backprop()!
	assert math.abs(prediction.grad.get_nth(0) - 0.25) < 1e-12
	assert math.abs(prediction.grad.get_nth(1) + 0.25) < 1e-12
}

fn test_cross_entropy_spatial_weighted_sum_ignores_indices_and_gradients() ! {
	ctx := autograd.ctx[f64]()
	prediction := ctx.variable(vtl.from_array([0.0, 10.0, math.log(3.0), 10.0], [1, 2, 1, 2])!)
	target := vtl.from_array([1, -100], [1, 1, 2])!
	weights := vtl.from_array([2.0, 4.0], [2])!
	mut result := cross_entropy_loss_indices[f64](weights, ignore_index: -100, reduction: 'sum').loss(prediction,
		target)!
	assert math.abs(result.value.get_nth(0) - 4.0 * math.log(4.0 / 3.0)) < 1e-12
	result.backprop()!
	for i, want in [1.0, 0.0, -1.0, 0.0] {
		assert math.abs(prediction.grad.get_nth(i) - want) < 1e-12
	}
}

fn test_cross_entropy_weighted_mean_uses_target_weight_sum() ! {
	ctx := autograd.ctx[f64]()
	logits := vtl.from_array([0.0, math.log(3.0), 0.0, math.log(3.0)], [2, 2])!
	prediction := ctx.variable(logits)
	target := vtl.from_array([0, 1], [2])!
	weights := vtl.from_array([1.0, 3.0], [2])!
	mut result := cross_entropy_loss_indices[f64](weights).loss(prediction, target)!
	want := (math.log(4.0) + 3.0 * math.log(4.0 / 3.0)) / 4.0
	assert math.abs(result.value.get_nth(0) - want) < 1e-12
	result.backprop()!
	for i, expected in [-0.1875, 0.1875, 0.140625, -0.140625] {
		assert math.abs(prediction.grad.get_nth(i) - expected) < 1e-12
	}
}

fn test_cross_entropy_soft_targets_with_default_ignore_index() ! {
	ctx := autograd.ctx[f64]()
	prediction := ctx.variable(vtl.from_array([0.0, math.log(3.0)], [2])!)
	target := vtl.from_array([0.25, 0.75], [2])!
	mut result := cross_entropy_loss[f64]().loss(prediction, target)!
	assert math.abs(result.value.get_nth(0) - (-0.25 * math.log(0.25) - 0.75 * math.log(0.75))) < 1e-12
	result.backprop()!
	assert math.abs(prediction.grad.get_nth(0)) < 1e-12
	assert math.abs(prediction.grad.get_nth(1)) < 1e-12
}
