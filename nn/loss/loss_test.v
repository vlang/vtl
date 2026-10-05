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
