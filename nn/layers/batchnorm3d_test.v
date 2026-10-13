module layers

import math
import vtl
import vtl.autograd

fn bn3d_input(c &autograd.Context[f64], arr []f64, shape []int) !&autograd.Variable[f64] {
	return c.variable(vtl.from_array(arr, shape)!)
}

fn test_batchnorm3d_training_normalizes_per_channel() ! {
	c := autograd.ctx[f64]()
	layer := batchnorm3d_layer[f64](c, 1, BatchNorm3DConfig{})
	input := bn3d_input(c, [1.0, 3.0, 5.0, 7.0], [2, 1, 1, 1, 2])!
	output := layer.forward_with_mode(input, true)!
	assert output.value.shape == [2, 1, 1, 1, 2]
	// Channel values [1,3,5,7]: mean 4, biased var 5, gamma=1, beta=0.
	inv_std := 1.0 / math.sqrt(5.0 + 1e-5)
	expected := [(1.0 - 4.0) * inv_std, (3.0 - 4.0) * inv_std, (5.0 - 4.0) * inv_std,
		(7.0 - 4.0) * inv_std]
	for i, want in expected {
		got := output.value.get_nth(i)
		assert math.abs(got - want) < 1e-12, 'output ${i}: got ${got}, want ${want}'
	}
	assert layer.variables().len == 2
	assert layer.output_shape() == [1]
}

fn test_batchnorm3d_running_stats_use_unbiased_variance() ! {
	c := autograd.ctx[f64]()
	mut layer := &BatchNorm3DLayer[f64]{
		eps:          1e-5
		momentum:     0.1
		affine:       true
		gamma:        c.variable(vtl.ones[f64]([1]))
		beta:         c.variable(vtl.zeros[f64]([1]))
		running_mean: vtl.zeros[f64]([1])
		running_var:  vtl.ones[f64]([1])
	}
	input := bn3d_input(c, [1.0, 3.0, 5.0, 7.0], [2, 1, 1, 1, 2])!
	layer.forward_with_mode(input, true)!

	// running_mean = 0.1 * 4; running_var = 0.9 * 1 + 0.1 * (20/3).
	assert math.abs(layer.running_mean.get_nth(0) - 0.4) < 1e-12
	assert math.abs(layer.running_var.get_nth(0) - (0.9 + 0.1 * 20.0 / 3.0)) < 1e-12

	// Eval mode normalizes with the running statistics.
	eval_input := bn3d_input(c, [4.0], [1, 1, 1, 1, 1])!
	eval_output := layer.forward_with_mode(eval_input, false)!
	expected := (4.0 - 0.4) / math.sqrt(0.9 + 0.1 * 20.0 / 3.0 + 1e-5)
	assert math.abs(eval_output.value.get_nth(0) - expected) < 1e-12
}

fn test_batchnorm3d_backward_matches_finite_differences() ! {
	c := autograd.ctx[f64]()
	layer := batchnorm3d_layer[f64](c, 1, BatchNorm3DConfig{})
	base := [1.0, 2.0, 3.0, 4.0]
	weights := c.variable(vtl.from_array([1.0, 0.5, -0.25, 2.0], [1, 1, 1, 2, 2])!,
		requires_grad: false)

	input := bn3d_input(c, base, [1, 1, 1, 2, 2])!
	output := layer.forward_with_mode(input, true)!
	mut loss := output.multiply(weights)!.sum()!
	loss.backprop()!

	h := 1e-6
	for i in 0 .. 4 {
		mut plus := base.clone()
		plus[i] += h
		plus_input := bn3d_input(c, plus, [1, 1, 1, 2, 2])!
		plus_out := layer.forward_with_mode(plus_input, true)!
		plus_loss := plus_out.multiply(weights)!.sum()!

		mut minus := base.clone()
		minus[i] -= h
		minus_input := bn3d_input(c, minus, [1, 1, 1, 2, 2])!
		minus_out := layer.forward_with_mode(minus_input, true)!
		minus_loss := minus_out.multiply(weights)!.sum()!

		numeric := (plus_loss.value.get_nth(0) - minus_loss.value.get_nth(0)) / (2.0 * h)
		analytic := input.grad.get_nth(i)
		assert math.abs(numeric - analytic) < 1e-4, 'grad ${i}: analytic ${analytic}, numeric ${numeric}'
	}
	// dbeta = sum of weights; dgamma = sum of weights * normalized input.
	vars := layer.variables()
	assert math.abs(vars[1].grad.get_nth(0) - 3.25) < 1e-12
	inv_std := 1.0 / math.sqrt(1.25 + 1e-5)
	want_dgamma := (1.0 * (1.0 - 2.5) + 0.5 * (2.0 - 2.5) - 0.25 * (3.0 - 2.5) + 2.0 * (4.0 - 2.5)) * inv_std
	assert math.abs(vars[0].grad.get_nth(0) - want_dgamma) < 1e-9
}

fn test_batchnorm3d_eval_backward_treats_running_stats_as_constant() ! {
	c := autograd.ctx[f64]()
	layer := batchnorm3d_layer[f64](c, 1, BatchNorm3DConfig{})
	input := bn3d_input(c, [1.0, 3.0], [1, 1, 1, 1, 2])!
	output := layer.forward_with_mode(input, false)!
	weights := c.variable(vtl.from_array([1.0, 0.0], [1, 1, 1, 1, 2])!, requires_grad: false)
	mut loss := output.multiply(weights)!.sum()!
	loss.backprop()!

	// Fresh layer: running_mean = 0, running_var = 1, gamma = 1.
	assert math.abs(input.grad.get_nth(0) - 1.0 / math.sqrt(1.0 + 1e-5)) < 1e-12
	assert input.grad.get_nth(1) == 0.0
}

fn test_batchnorm3d_rejects_wrong_rank_and_channels() ! {
	c := autograd.ctx[f64]()
	layer := batchnorm3d_layer[f64](c, 2, BatchNorm3DConfig{})
	bad_rank := bn3d_input(c, [1.0, 2.0], [1, 2])!
	if _ := layer.forward_with_mode(bad_rank, true) {
		assert false, 'expected rank error'
	}
	bad_channels := bn3d_input(c, [1.0, 2.0, 3.0], [1, 3, 1, 1, 1])!
	if _ := layer.forward_with_mode(bad_channels, true) {
		assert false, 'expected channel error'
	}
	no_affine := batchnorm3d_layer[f64](c, 2, BatchNorm3DConfig{
		affine: false
	})
	assert no_affine.variables().len == 0
}
