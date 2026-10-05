module optimizers

import math
import vtl
import vtl.autograd

fn test_clip_grad_norm_scales_global_norm() ! {
	context := autograd.ctx[f64]()
	first := context.variable(vtl.from_1d([0.0, 0.0])!)
	second := context.variable(vtl.from_1d([0.0])!)
	first.grad.set_nth(0, 3.0)
	first.grad.set_nth(1, 4.0)
	second.grad.set_nth(0, 0.0)
	mut parameters := [first, second]

	original_norm := clip_grad_norm[f64](mut parameters, 2.0)!

	assert math.abs(original_norm - 5.0) < 1e-12
	assert math.abs(first.grad.get_nth(0) - 1.2) < 1e-12
	assert math.abs(first.grad.get_nth(1) - 1.6) < 1e-12
	assert second.grad.get_nth(0) == 0.0
}

fn test_clip_grad_norm_preserves_small_and_f32_gradients() ! {
	context := autograd.ctx[f32]()
	parameter := context.variable(vtl.from_1d([0.3, 0.4])!)
	parameter.grad.set_nth(0, f32(0.3))
	parameter.grad.set_nth(1, f32(0.4))
	mut parameters := [parameter]
	before := clip_grad_norm[f32](mut parameters, 1.0)!
	assert math.abs(before - 0.5) < 1e-6
	assert math.abs(f64(parameter.grad.get_nth(0)) - 0.3) < 1e-6
	assert math.abs(f64(parameter.grad.get_nth(1)) - 0.4) < 1e-6

	parameter.grad.set_nth(0, f32(6.0))
	parameter.grad.set_nth(1, f32(8.0))
	clipped_norm := clip_grad_norm[f32](mut parameters, 5.0)!
	assert math.abs(clipped_norm - 10.0) < 1e-6
	assert math.abs(f64(parameter.grad.get_nth(0)) - 3.0) < 1e-6
	assert math.abs(f64(parameter.grad.get_nth(1)) - 4.0) < 1e-6
}

fn test_clip_grad_norm_ignores_frozen_parameters() ! {
	context := autograd.ctx[f64]()
	trainable := context.variable(vtl.from_1d([0.0])!)
	frozen := context.variable(vtl.from_1d([0.0])!, requires_grad: false)
	trainable.grad.set_nth(0, 6.0)
	frozen.grad.set_nth(0, 8.0)
	mut parameters := [trainable, frozen]

	original_norm := clip_grad_norm[f64](mut parameters, 3.0)!

	assert math.abs(original_norm - 6.0) < 1e-12
	assert trainable.grad.get_nth(0) == 3.0
	assert frozen.grad.get_nth(0) == 8.0
}

fn test_clip_grad_norm_rejects_invalid_limit() {
	context := autograd.ctx[f64]()
	parameter := context.variable(vtl.from_1d([0.0])!)
	mut parameters := [parameter]
	_ := clip_grad_norm[f64](mut parameters, 0.0) or {
		assert err.msg().contains('max_norm must be finite and greater than zero')
		return
	}
	assert false, 'expected clip_grad_norm to reject a zero limit'
}
