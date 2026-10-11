module loss

import vtl
import vtl.autograd
import math

fn ctx[T]() &autograd.Context[T] {
	return autograd.ctx[T]()
}

fn variable[T](c &autograd.Context[T], arr []T, shape []int) !&autograd.Variable[T] {
	t := vtl.from_array(arr, shape)!
	return c.variable(t)
}

fn tensor[T](arr []T, shape []int) !&vtl.Tensor[T] {
	return vtl.from_array(arr, shape)!
}

// MSE loss: mean((pred - target)^2)
fn test_mse_loss_forward() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [1.0, 2.0, 3.0, 4.0], [2, 2])!
	target := tensor[f64]([1.0, 2.0, 3.0, 4.0], [2, 2])!
	l := mse_loss[f64]()
	result := l.loss(pred, target)!
	v := result.value.get_nth(0)
	assert v < 1e-9, 'MSE of identical tensors should be 0, got ${v}'
}

fn test_mse_loss_nonzero() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [2.0, 2.0], [1, 2])!
	target := tensor[f64]([0.0, 0.0], [1, 2])!
	l := mse_loss[f64]()
	result := l.loss(pred, target)!
	v := result.value.get_nth(0)
	// mean((2-0)^2 + (2-0)^2) / 2 = 4.0
	assert v - 4.0 < 1e-6 && v - 4.0 > -1e-6, 'MSE expected 4.0, got ${v}'
}

fn test_mse_loss_backward() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [3.0, 1.0], [1, 2])!
	target := tensor[f64]([1.0, 1.0], [1, 2])!
	l := mse_loss[f64]()
	mut result := l.loss(pred, target)!
	result.backprop()!
	// grad w.r.t pred[0] = 2*(3-1)/2 = 2.0
	g := pred.grad.get_nth(0)
	assert g - 2.0 < 1e-6 && g - 2.0 > -1e-6, 'MSE backward grad[0] expected 2.0, got ${g}'
}

fn test_mse_loss_backward_uses_mean_reduction_and_factor_two() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [1.0, 2.0, 3.0, 4.0], [2, 2])!
	target := tensor[f64]([0.0, 0.0, 0.0, 0.0], [2, 2])!
	mut result := mse_loss[f64]().loss(pred, target)!
	result.backprop()!
	for i, want in [0.5, 1.0, 1.5, 2.0] {
		assert math.abs(pred.grad.get_nth(i) - want) < 1e-12
	}
}

fn test_cross_entropy_loss_backward_uses_logits_and_target_mass() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, math.log(3.0)], [1, 2])!
	target := tensor[f64]([1.0, 1.0], [1, 2])!
	mut result := cross_entropy_loss[f64]().loss(pred, target)!
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0) + 0.5) < 1e-12
	assert math.abs(pred.grad.get_nth(1) - 0.5) < 1e-12
}

fn test_cross_entropy_loss_supports_class_index_targets() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, math.log(3.0)], [1, 2])!
	target := tensor[f64]([1.0], [1])!
	mut result := cross_entropy_loss[f64]().loss(pred, target)!
	assert math.abs(result.value.get_nth(0) - math.log(4.0 / 3.0)) < 1e-12
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0) - 0.25) < 1e-12
	assert math.abs(pred.grad.get_nth(1) + 0.25) < 1e-12
}

fn test_cross_entropy_loss_rejects_invalid_class_index_targets() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, math.log(3.0)], [1, 2])!
	for invalid_class in [1.5, 2.0, -1.0] {
		target := tensor[f64]([invalid_class], [1])!
		cross_entropy_loss[f64]().loss(pred, target) or {
			assert err.msg().contains('class index')
			continue
		}
		assert false, 'Cross entropy should reject invalid class index ${invalid_class}'
	}
}

fn test_softmax_cross_entropy_loss_backward_uses_target_mass() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, math.log(3.0)], [1, 2])!
	target := tensor[f64]([1.0, 1.0], [1, 2])!
	mut result := softmax_cross_entropy_loss[f64]().loss(pred, target)!
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0) + 0.5) < 1e-12
	assert math.abs(pred.grad.get_nth(1) - 0.5) < 1e-12
}

// BCE loss tests (from_logits=false → raw probabilities)
fn test_bce_loss_forward() ! {
	c := ctx[f64]()
	// use from_logits: false so input is raw probability
	pred := variable[f64](c, [0.9, 0.1], [1, 2])!
	target := tensor[f64]([1.0, 0.0], [1, 2])!
	l := bce_loss[f64](from_logits: false)
	result := l.loss(pred, target)!
	v := result.value.get_nth(0)
	assert v > 0.0, 'BCE loss should be positive, got ${v}'
	assert v < 0.5, 'BCE loss for near-correct prediction should be small, got ${v}'
}

fn test_bce_loss_logits_forward_matches_stable_reference() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, math.log(3.0)], [1, 2])!
	target := tensor[f64]([1.0, 0.0], [1, 2])!
	result := bce_loss[f64]().loss(pred, target)!
	want := (math.log(2.0) + math.log(4.0)) / 2.0
	assert math.abs(result.value.get_nth(0) - want) < 1e-12
}

fn test_bce_loss_logits_forward_preserves_extreme_penalties() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [1000.0, -1000.0], [2])!
	target := tensor[f64]([0.0, 1.0], [2])!
	result := bce_loss[f64]().loss(pred, target)!
	assert math.abs(result.value.get_nth(0) - 1000.0) < 1e-12
}

fn test_bce_loss_logits_backward_matches_mean_bce() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, math.log(3.0)], [1, 2])!
	target := tensor[f64]([1.0, 0.0], [1, 2])!
	mut result := bce_loss[f64]().loss(pred, target)!
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0) + 0.25) < 1e-12
	assert math.abs(pred.grad.get_nth(1) - 0.375) < 1e-12
}

fn test_bce_loss_probability_backward_matches_mean_bce() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.25, 0.75], [2, 1])!
	target := tensor[f64]([1.0, 0.0], [2, 1])!
	mut result := bce_loss[f64](from_logits: false).loss(pred, target)!
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0) + 2.0) < 1e-12
	assert math.abs(pred.grad.get_nth(1) - 2.0) < 1e-12
}

fn test_sigmoid_cross_entropy_loss_backward_matches_batch_mean() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, math.log(3.0)], [2, 1])!
	target := tensor[f64]([1.0, 0.0], [2])!
	mut result := sigmoid_cross_entropy_loss[f64]().loss(pred, target)!
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0) + 0.25) < 1e-12
	assert math.abs(pred.grad.get_nth(1) - 0.375) < 1e-12
}

fn test_bce_loss_clamps_logarithms_at_minus_100() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, 1.0], [2])!
	target := tensor[f64]([1.0, 0.0], [2])!
	result := bce_loss[f64](from_logits: false).loss(pred, target)!
	assert math.abs(result.value.get_nth(0) - 100.0) < 1e-12
}

fn test_bce_loss_probability_correct_boundaries_have_zero_loss() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, 1.0], [2])!
	target := tensor[f64]([0.0, 1.0], [2])!
	result := bce_loss[f64](from_logits: false).loss(pred, target)!
	assert result.value.get_nth(0) == 0.0
}

fn test_bce_loss_probability_rejects_values_outside_unit_interval() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [-0.1], [1])!
	target := tensor[f64]([1.0], [1])!
	bce_loss[f64](from_logits: false).loss(pred, target) or {
		assert err.msg().contains('probabilities must be in [0, 1]')
		return
	}
	assert false, 'BCE loss should reject probabilities outside [0, 1]'
}

fn test_bce_loss_strided_inputs_use_strided_path() ! {
	c := ctx[f64]()
	input_base := tensor[f64]([0.1, 0.2, 0.7, 0.8], [2, 2])!
	target_base := tensor[f64]([0.0, 1.0, 1.0, 0.0], [2, 2])!
	pred := c.variable(input_base.transpose([1, 0])!)
	target := target_base.transpose([1, 0])!
	result := bce_loss[f64](from_logits: false).loss(pred, target)!
	expected := (-math.log(0.9) - math.log(0.7) - math.log(0.2) - math.log(0.2)) / 4.0
	assert math.abs(result.value.get_nth(0) - expected) < 1e-12
}

fn test_bce_loss_rejects_broadcast_shapes() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.2, 0.8], [2])!
	target := tensor[f64]([1.0], [1])!
	bce_loss[f64]().loss(pred, target) or {
		assert err.msg().contains('identical shapes')
		return
	}
	assert false, 'BCE loss should reject mismatched input and target shapes'
}

// Huber loss tests
fn test_huber_loss_forward_small_error() ! {
	c := ctx[f64]()
	// |error| < delta → quadratic: 0.5 * err^2
	pred := variable[f64](c, [1.5], [1, 1])!
	target := tensor[f64]([1.0], [1, 1])!
	l := huber_loss[f64](delta: 1.0)
	result := l.loss(pred, target)!
	v := result.value.get_nth(0)
	// 0.5 * (0.5)^2 = 0.125
	assert v - 0.125 < 1e-6 && v - 0.125 > -1e-6, 'Huber small-error expected 0.125, got ${v}'
}

fn test_huber_loss_forward_large_error() ! {
	c := ctx[f64]()
	// |error| >= delta → linear: delta*(|err| - 0.5*delta)
	pred := variable[f64](c, [3.0], [1, 1])!
	target := tensor[f64]([0.0], [1, 1])!
	l := huber_loss[f64](delta: 1.0)
	result := l.loss(pred, target)!
	v := result.value.get_nth(0)
	// 1.0*(3.0 - 0.5) = 2.5
	assert v - 2.5 < 1e-6 && v - 2.5 > -1e-6, 'Huber large-error expected 2.5, got ${v}'
}

fn test_huber_loss_backward_uses_input_and_mean_reduction() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [1.5, 3.0], [2])!
	target := tensor[f64]([1.0, 0.0], [2])!
	mut result := huber_loss[f64](delta: 1.0).loss(pred, target)!
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0) - 0.25) < 1e-12
	assert math.abs(pred.grad.get_nth(1) - 0.5) < 1e-12
}

fn test_nll_loss_backward_uses_log_probability_derivative() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [-2.0, -0.5], [1, 2])!
	target := tensor[f64]([1.0, 0.0], [1, 2])!
	mut result := nll_loss[f64](unsafe { nil }).loss(pred, target)!
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0) + 1.0) < 1e-12
	assert math.abs(pred.grad.get_nth(1)) < 1e-12
}

fn test_nll_loss_supports_class_index_targets() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [-2.0, -0.5, -0.3, -1.2], [2, 2])!
	target := tensor[f64]([1.0, 0.0], [2])!
	mut result := nll_loss[f64](unsafe { nil }).loss(pred, target)!
	assert math.abs(result.value.get_nth(0) - 0.4) < 1e-12
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0)) < 1e-12
	assert math.abs(pred.grad.get_nth(1) + 0.5) < 1e-12
	assert math.abs(pred.grad.get_nth(2) + 0.5) < 1e-12
	assert math.abs(pred.grad.get_nth(3)) < 1e-12
}

fn test_nll_loss_applies_class_weights_to_index_targets() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [-2.0, -0.5, -0.3, -1.2], [2, 2])!
	target := tensor[f64]([1.0, 0.0], [2])!
	weights := tensor[f64]([2.0, 4.0], [2])!
	mut result := nll_loss[f64](weights).loss(pred, target)!
	assert math.abs(result.value.get_nth(0) - (2.6 / 6.0)) < 1e-12
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0)) < 1e-12
	assert math.abs(pred.grad.get_nth(1) + (4.0 / 6.0)) < 1e-12
	assert math.abs(pred.grad.get_nth(2) + (2.0 / 6.0)) < 1e-12
	assert math.abs(pred.grad.get_nth(3)) < 1e-12
}

fn test_nll_loss_applies_class_weights_to_soft_targets() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [-2.0, -0.5, -0.3, -1.2], [2, 2])!
	target := tensor[f64]([0.0, 1.0, 1.0, 0.0], [2, 2])!
	weights := tensor[f64]([2.0, 4.0], [2])!
	mut result := nll_loss[f64](weights).loss(pred, target)!
	assert math.abs(result.value.get_nth(0) - 1.3) < 1e-12
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0)) < 1e-12
	assert math.abs(pred.grad.get_nth(1) + 2.0) < 1e-12
	assert math.abs(pred.grad.get_nth(2) + 1.0) < 1e-12
	assert math.abs(pred.grad.get_nth(3)) < 1e-12
}

fn test_nll_loss_sum_reduction_and_ignore_index() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [-2.0, -0.5, -0.3, -1.2], [2, 2])!
	target := tensor[f64]([1.0, -100.0], [2])!
	weights := tensor[f64]([2.0, 4.0], [2])!
	mut result := nll_loss_with_options[f64](weights, ignore_index: -100, reduction: 'sum').loss(pred,
		target)!
	assert math.abs(result.value.get_nth(0) - 2.0) < 1e-12
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0)) < 1e-12
	assert math.abs(pred.grad.get_nth(1) + 4.0) < 1e-12
	assert math.abs(pred.grad.get_nth(2)) < 1e-12
	assert math.abs(pred.grad.get_nth(3)) < 1e-12
}

fn test_nll_loss_mean_ignores_target_in_denominator_and_gradient() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [-2.0, -0.5, -0.3, -1.2], [2, 2])!
	target := tensor[f64]([1.0, -100.0], [2])!
	mut result := nll_loss_with_options[f64](unsafe { nil }, ignore_index: -100).loss(pred, target)!
	assert math.abs(result.value.get_nth(0) - 0.5) < 1e-12
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0)) < 1e-12
	assert math.abs(pred.grad.get_nth(1) + 1.0) < 1e-12
	assert math.abs(pred.grad.get_nth(2)) < 1e-12
	assert math.abs(pred.grad.get_nth(3)) < 1e-12
}

fn test_nll_loss_none_reduction_returns_per_sample_losses_and_gradients() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [-2.0, -0.5, -0.3, -1.2], [2, 2])!
	target := tensor[f64]([1.0, -100.0], [2])!
	weights := tensor[f64]([2.0, 4.0], [2])!
	mut result := nll_loss_with_options[f64](weights, ignore_index: -100, reduction: 'none').loss(pred,
		target)!
	assert result.value.shape == [2]
	assert math.abs(result.value.get_nth(0) - 2.0) < 1e-12
	assert math.abs(result.value.get_nth(1)) < 1e-12
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0)) < 1e-12
	assert math.abs(pred.grad.get_nth(1) + 4.0) < 1e-12
	assert math.abs(pred.grad.get_nth(2)) < 1e-12
	assert math.abs(pred.grad.get_nth(3)) < 1e-12
}

fn test_kl_div_loss_backward_is_negative_target_mean() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [-2.0, -0.5], [2])!
	target := tensor[f64]([0.25, 0.75], [2])!
	mut result := kl_div_loss[f64]().loss(pred, target)!
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0) + 0.125) < 1e-12
	assert math.abs(pred.grad.get_nth(1) + 0.375) < 1e-12
}

fn test_l1_loss_forward_and_backward() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [1.0, 3.0], [2])!
	target := tensor[f64]([0.0, 4.0], [2])!
	mut result := l1_loss[f64]().loss(pred, target)!
	assert math.abs(result.value.get_nth(0) - 1.0) < 1e-12
	result.backprop()!
	assert math.abs(pred.grad.get_nth(0) - 0.5) < 1e-12
	assert math.abs(pred.grad.get_nth(1) + 0.5) < 1e-12
}

fn test_hinge_loss_forward_and_backward() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [2.0, -0.5], [2])!
	target := tensor[f64]([1.0, -1.0], [2])!
	mut result := hinge_loss[f64]().loss(pred, target)!
	assert math.abs(result.value.get_nth(0) - 0.25) < 1e-12
	result.backprop()!
	assert pred.grad.get_nth(0) == 0.0
	assert math.abs(pred.grad.get_nth(1) - 0.5) < 1e-12
}

fn test_focal_loss_forward_and_backward_logits() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0], [1])!
	target := tensor[f64]([1.0], [1])!
	mut result := focal_loss[f64]().loss(pred, target)!
	assert math.abs(result.value.get_nth(0) - 0.25 * 0.25 * math.log(2.0)) < 1e-12
	result.backprop()!
	want_grad := 0.25 * (2.0 * 0.5 * math.log(0.5) - 0.25 / 0.5) * 0.25
	assert math.abs(pred.grad.get_nth(0) - want_grad) < 1e-12
}

fn test_l1_loss_rejects_incompatible_shapes() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, 1.0], [2])!
	bad_target := tensor[f64]([1.0], [1])!
	l1_loss[f64]().loss(pred, bad_target) or {
		assert err.msg().contains('identical shapes')
		return
	}
	assert false, 'L1 loss should reject incompatible shapes'
}

fn test_hinge_loss_rejects_invalid_labels() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, 1.0], [2])!
	target := tensor[f64]([0.0, 1.0], [2])!
	hinge_loss[f64]().loss(pred, target) or {
		assert err.msg().contains('labels must be -1 or +1')
		return
	}
	assert false, 'hinge loss should reject invalid labels'
}

fn test_focal_loss_rejects_invalid_targets() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0, 1.0], [2])!
	target := tensor[f64]([1.0, 2.0], [2])!
	focal_loss[f64]().loss(pred, target) or {
		assert err.msg().contains('targets must be 0 or +1')
		return
	}
	assert false, 'focal loss should reject invalid targets'
}

fn test_focal_loss_probability_inputs() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.8], [1])!
	target := tensor[f64]([1.0], [1])!
	mut result := focal_loss[f64](from_logits: false, alpha: 0.25, gamma: 2.0).loss(pred, target)!
	want := -0.25 * math.pow(0.2, 2.0) * math.log(0.8)
	assert math.abs(result.value.get_nth(0) - want) < 1e-12
	result.backprop()!
	want_grad := 0.25 * (2.0 * 0.2 * math.log(0.8) - math.pow(0.2, 2.0) / 0.8)
	assert math.abs(pred.grad.get_nth(0) - want_grad) < 1e-12
}

fn test_focal_loss_negative_class_gradient() ! {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0], [1])!
	target := tensor[f64]([0.0], [1])!
	mut result := focal_loss[f64]().loss(pred, target)!
	want := 0.75 * 0.25 * math.log(2.0)
	assert math.abs(result.value.get_nth(0) - want) < 1e-12
	result.backprop()!
	want_grad := 0.75 * (2.0 * 0.5 * math.log(0.5) - 0.25 / 0.5) * -0.25
	assert math.abs(pred.grad.get_nth(0) - want_grad) < 1e-12
}

fn test_focal_loss_rejects_non_finite_parameters() {
	c := ctx[f64]()
	pred := variable[f64](c, [0.0], [1])!
	target := tensor[f64]([1.0], [1])!
	for alpha in [math.nan(), math.inf(1)] {
		focal_loss[f64](alpha: alpha, gamma: 2.0).loss(pred, target) or { continue }
		assert false, 'focal loss should reject non-finite alpha'
	}
	for gamma in [math.nan(), math.inf(1)] {
		focal_loss[f64](alpha: 0.25, gamma: gamma).loss(pred, target) or { continue }
		assert false, 'focal loss should reject non-finite gamma'
	}
}
