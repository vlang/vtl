module internal

import math
import vtl

fn layer_norm_test_loss(input_values []f64, gamma_values []f64, beta_values []f64, gradient_values []f64) !f64 {
	input := vtl.from_array(input_values, [2, 3])!
	gamma := vtl.from_array(gamma_values, [3])!
	beta := vtl.from_array(beta_values, [3])!
	gradient := vtl.from_array(gradient_values, [2, 3])!
	output := layer_norm_forward_shape[f64](input, gamma, beta, [3], 1e-5)!
	mut loss := 0.0
	for i in 0 .. output.size {
		loss += output.get_nth(i) * gradient.get_nth(i)
	}
	return loss
}

fn test_layer_norm_forward_normalizes_each_trailing_block_and_broadcasts_affine_parameters() ! {
	input := vtl.from_array([1.0, 2.0, 4.0, 0.0, 3.0, 8.0], [2, 3])!
	gamma := vtl.from_array([1.5, 0.5, 2.0], [3])!
	beta := vtl.from_array([0.1, -0.2, 0.3], [3])!
	output := layer_norm_forward_shape[f64](input, gamma, beta, [3], 1e-5)!

	for start in [0, 3] {
		mut mean := 0.0
		for i in start .. start + 3 {
			mean += input.get_nth(i)
		}
		mean /= 3.0
		mut variance := 0.0
		for i in start .. start + 3 {
			diff := input.get_nth(i) - mean
			variance += diff * diff
		}
		inv_std := 1.0 / math.sqrt(variance / 3.0 + 1e-5)
		for i in 0 .. 3 {
			expected := (input.get_nth(start + i) - mean) * inv_std * gamma.get_nth(i) + beta.get_nth(i)
			assert math.abs(output.get_nth(start + i) - expected) < 1e-12
		}
	}
}

fn test_layer_norm_backward_matches_finite_differences_for_input_and_affine_parameters() ! {
	input_values := [1.0, 2.0, 4.0, 0.0, 3.0, 8.0]
	gamma_values := [1.5, 0.5, 2.0]
	beta_values := [0.1, -0.2, 0.3]
	gradient_values := [0.3, -0.7, 1.1, 1.2, 0.4, -0.2]
	input := vtl.from_array(input_values, [2, 3])!
	gamma := vtl.from_array(gamma_values, [3])!
	beta := vtl.from_array(beta_values, [3])!
	gradient := vtl.from_array(gradient_values, [2, 3])!
	grads := layer_norm_backward_shape[f64](gradient, input, gamma, beta, [3], 1e-5)!
	step := 1e-6

	for i in 0 .. input_values.len {
		mut plus := input_values.clone()
		mut minus := input_values.clone()
		plus[i] += step
		minus[i] -= step
		numeric := (layer_norm_test_loss(plus, gamma_values, beta_values, gradient_values)! - layer_norm_test_loss(minus,
			gamma_values, beta_values, gradient_values)!) / (2.0 * step)
		assert math.abs(grads[0].get_nth(i) - numeric) < 2e-6
	}

	for i in 0 .. gamma_values.len {
		mut plus := gamma_values.clone()
		mut minus := gamma_values.clone()
		plus[i] += step
		minus[i] -= step
		numeric := (layer_norm_test_loss(input_values, plus, beta_values, gradient_values)! - layer_norm_test_loss(input_values,
			minus, beta_values, gradient_values)!) / (2.0 * step)
		assert math.abs(grads[1].get_nth(i) - numeric) < 2e-6
	}

	for i in 0 .. beta_values.len {
		mut plus := beta_values.clone()
		mut minus := beta_values.clone()
		plus[i] += step
		minus[i] -= step
		numeric := (layer_norm_test_loss(input_values, gamma_values, plus, gradient_values)! - layer_norm_test_loss(input_values,
			gamma_values, minus, gradient_values)!) / (2.0 * step)
		assert math.abs(grads[2].get_nth(i) - numeric) < 2e-6
	}
}

fn test_layer_norm_rejects_mismatched_trailing_or_affine_shapes() ! {
	input := vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!
	wrong_gamma := vtl.ones[f64]([4])
	if _ := layer_norm_forward_shape[f64](input, wrong_gamma, unsafe { nil }, [2], 1e-5) {
		assert false, 'expected mismatched gamma shape to fail'
	}
	if _ := layer_norm_forward_shape[f64](input, unsafe { nil }, unsafe { nil }, [3], 1e-5) {
		assert false, 'expected mismatched normalized shape to fail'
	}
}

fn test_layer_norm_backward_returns_each_individual_affine_gradient() ! {
	input := vtl.from_array([1.0, 2.0, 4.0, 0.0, 3.0, 8.0], [2, 3])!
	gradient := vtl.from_array([0.3, -0.7, 1.1, 1.2, 0.4, -0.2], [2, 3])!
	gamma := vtl.from_array([1.5, 0.5, 2.0], [3])!
	beta := vtl.from_array([0.1, -0.2, 0.3], [3])!

	gamma_only := layer_norm_backward_shape[f64](gradient, input, gamma, unsafe { nil }, [3], 1e-5)!
	assert gamma_only.len == 2
	assert gamma_only[1].shape == [3]

	beta_only := layer_norm_backward_shape[f64](gradient, input, unsafe { nil }, beta, [3], 1e-5)!
	assert beta_only.len == 2
	assert beta_only[1].shape == [3]
	assert math.abs(beta_only[1].get_nth(0) - 1.5) < 1e-12
	assert math.abs(beta_only[1].get_nth(1) - (-0.3)) < 1e-12
	assert math.abs(beta_only[1].get_nth(2) - 0.9) < 1e-12
}
