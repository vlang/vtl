module hooks

import math
import vtl
import vtl.autograd
import vtl.nn.layers

// Tests for module hooks (issue #808). They cover the ordering and replacement contract,
// removal, backward hooks with and without gradients, and hook errors.

fn scale_values(input &autograd.Variable[f64], factor f64, shift f64) !&autograd.Variable[f64] {
	values := input.value.to_array().map(it * factor + shift)
	return input.context.variable(vtl.from_array(values, input.value.shape)!)
}

fn pre_double(input &autograd.Variable[f64]) !&autograd.Variable[f64] {
	return scale_values(input, 2, 0)
}

fn pre_add_one(input &autograd.Variable[f64]) !&autograd.Variable[f64] {
	return scale_values(input, 1, 1)
}

fn post_triple(_input &autograd.Variable[f64], output &autograd.Variable[f64]) !&autograd.Variable[f64] {
	return scale_values(output, 3, 0)
}

fn post_subtract_two(_input &autograd.Variable[f64], output &autograd.Variable[f64]) !&autograd.Variable[f64] {
	return scale_values(output, 1, -2)
}

fn grad_double(grad &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return vtl.from_array(grad.to_array().map(it * 2), grad.shape)!
}

fn fail_pre(_input &autograd.Variable[f64]) !&autograd.Variable[f64] {
	return error('pre-hook rejected the input')
}

fn fail_backward(_grad &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return error('backward hook rejected the gradient')
}

fn assert_values_close(got []f64, want []f64, label string) {
	assert got.len == want.len, '${label}: length ${got.len} != ${want.len}'
	for i in 0 .. got.len {
		assert math.abs(got[i] - want[i]) < 1e-12, '${label}[${i}]: got ${got[i]}, want ${want[i]}'
	}
}

fn hooks_test_input(ctx &autograd.Context[f64]) &autograd.Variable[f64] {
	return ctx.variable(vtl.from_array([f64(-1), 0.5, 2, -3], [4]) or { panic(err) })
}

fn test_forward_hooks_run_in_registration_order_around_the_layer() ! {
	ctx := autograd.ctx[f64]()
	wrapped, hooked := with_hooks[f64](layers.relu_layer[f64](ctx, [4]))
	hooked.add_forward_pre_hook(pre_double)
	hooked.add_forward_pre_hook(pre_add_one)
	hooked.add_forward_hook(post_triple)
	hooked.add_forward_hook(post_subtract_two)

	input := hooks_test_input(ctx)
	output := wrapped.forward(input)!
	// Pre-hooks: x -> 2x -> 2x + 1. Relu. Post-hooks: y -> 3y -> 3y - 2.
	want := [-1.0, 0.5, 2, -3].map(fn (x f64) f64 {
		shifted := 2 * x + 1
		relu := if shifted > 0 { shifted } else { 0 }
		return 3 * relu - 2
	})
	assert_values_close(output.value.to_array(), want, 'forward order')
}

fn test_pre_hook_replaces_the_input_the_layer_sees() ! {
	ctx := autograd.ctx[f64]()
	wrapped, hooked := with_hooks[f64](layers.relu_layer[f64](ctx, [4]))
	hooked.add_forward_pre_hook(pre_add_one)
	output := wrapped.forward(hooks_test_input(ctx))!
	// x + 1 for [-1, 0.5, 2, -3] is [0, 1.5, 3, -2]; relu gives [0, 1.5, 3, 0].
	assert_values_close(output.value.to_array(), [f64(0), 1.5, 3, 0], 'pre-hook replacement')
}

fn test_post_hook_sees_the_input_and_can_replace_the_output() ! {
	ctx := autograd.ctx[f64]()
	wrapped, hooked := with_hooks[f64](layers.relu_layer[f64](ctx, [4]))
	hooked.add_forward_hook(fn (input &autograd.Variable[f64], output &autograd.Variable[f64]) !&autograd.Variable[f64] {
		// The hook sees the input the layer saw (unchanged here) and the layer output.
		assert input.value.to_array() == [f64(-1), 0.5, 2, -3]
		return scale_values(output, 0, 7)
	})
	output := wrapped.forward(hooks_test_input(ctx))!
	assert_values_close(output.value.to_array(), [f64(7), 7, 7, 7], 'post-hook replacement')
}

fn test_removed_hooks_no_longer_run() ! {
	ctx := autograd.ctx[f64]()
	wrapped, hooked := with_hooks[f64](layers.relu_layer[f64](ctx, [4]))
	handle := hooked.add_forward_pre_hook(pre_double)
	before := wrapped.forward(hooks_test_input(ctx))!.value.to_array()
	handle.remove()
	handle.remove() // second removal is a no-op
	after := wrapped.forward(hooks_test_input(ctx))!.value.to_array()
	assert_values_close(before, [f64(0), 1, 4, 0], 'with pre-hook')
	assert_values_close(after, [f64(0), 0.5, 2, 0], 'after removal')
}

fn test_hook_errors_stop_the_forward_pass() {
	ctx := autograd.ctx[f64]()
	wrapped, hooked := with_hooks[f64](layers.relu_layer[f64](ctx, [4]))
	hooked.add_forward_pre_hook(fail_pre)
	if _ := wrapped.forward(hooks_test_input(ctx)) {
		assert false, 'a failing pre-hook must stop the forward pass'
	} else {
		assert err.msg() == 'pre-hook rejected the input'
	}
}

fn test_backward_hook_scales_the_gradient_and_removal_restores_it() ! {
	// Without hooks, grad of sum(relu(x)) is the relu indicator.
	ctx := autograd.ctx[f64]()
	plain_wrapped, _ := with_hooks[f64](layers.relu_layer[f64](ctx, [4]))
	plain_input := hooks_test_input(ctx)
	plain := plain_wrapped.forward(plain_input)!
	plain.backprop()!
	plain_grad := plain_input.grad.to_array()

	ctx2 := autograd.ctx[f64]()
	wrapped, hooked := with_hooks[f64](layers.relu_layer[f64](ctx2, [4]))
	handle := hooked.add_backward_hook(grad_double)
	input := hooks_test_input(ctx2)
	output := wrapped.forward(input)!
	output.backprop()!
	hooked_grad := input.grad.to_array()
	assert_values_close(hooked_grad, plain_grad.map(it * 2), 'backward hook doubles the gradient')

	handle.remove()
	ctx3 := autograd.ctx[f64]()
	again, hooked_again := with_hooks[f64](layers.relu_layer[f64](ctx3, [4]))
	handle_again := hooked_again.add_backward_hook(grad_double)
	handle_again.remove()
	again_input := hooks_test_input(ctx3)
	again_out := again.forward(again_input)!
	again_out.backprop()!
	assert_values_close(again_input.grad.to_array(), plain_grad, 'removed backward hook')
}

fn test_backward_hook_error_stops_backpropagation() ! {
	ctx := autograd.ctx[f64]()
	wrapped, hooked := with_hooks[f64](layers.relu_layer[f64](ctx, [4]))
	hooked.add_backward_hook(fail_backward)
	input := hooks_test_input(ctx)
	output := wrapped.forward(input)!
	if _ := output.backprop() {
		assert false, 'a failing backward hook must stop backpropagation'
	} else {
		assert err.msg() == 'backward hook rejected the gradient'
	}
}

// Without gradients, backward hooks are not attached: the output does not require a
// gradient, so no backward node exists and the hook cannot run.
fn test_backward_hooks_are_not_attached_without_gradients() ! {
	mut ctx := autograd.ctx[f64]()
	wrapped, hooked := with_hooks[f64](layers.relu_layer[f64](ctx, [4]))
	hooked.add_backward_hook(fail_backward)
	ctx.no_grad = true
	output := wrapped.forward(hooks_test_input(ctx))!
	assert !output.is_grad_needed()
	assert_values_close(output.value.to_array(), [f64(0), 0.5, 2, 0], 'no-grad forward still runs')
	// The backward hook was not attached, so backprop on this output has no hook to fail on.
	output.backprop() or { assert false, 'no-grad output must not carry the backward hook: ${err}' }
}

fn test_wrapper_exposes_the_layer_variables_unchanged() {
	ctx := autograd.ctx[f64]()
	layer := layers.linear_layer[f64](ctx, 3, 2)
	wrapped, _ := with_hooks[f64](layer)
	assert wrapped.variables().len == layer.variables().len
	assert wrapped.output_shape() == layer.output_shape()
}
