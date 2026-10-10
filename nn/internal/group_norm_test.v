module internal

import math
import vtl

fn group_norm_test_loss(input_values []f64, gamma_values []f64, beta_values []f64, gradient_values []f64) !f64 {
	input := vtl.from_array(input_values, [1, 2, 2])!
	gamma := vtl.from_array(gamma_values, [2])!
	beta := vtl.from_array(beta_values, [2])!
	gradient := vtl.from_array(gradient_values, [1, 2, 2])!
	output := group_norm_forward[f64](input, 1, gamma, beta, 1e-5)!
	mut loss := 0.0
	for i in 0 .. output.size {
		loss += output.get_nth(i) * gradient.get_nth(i)
	}
	return loss
}

fn test_group_norm_forward_normalizes_channel_groups_across_spatial_dimensions() ! {
	input := vtl.from_array([1.0, 2.0, 4.0, 0.0, 3.0, 8.0, 5.0, 7.0], [1, 4, 2])!
	output := group_norm_forward[f64](input, 2, unsafe { nil }, unsafe { nil }, 1e-5)!
	for group_start in [0, 4] {
		mut mean := 0.0
		for i in group_start .. group_start + 4 {
			mean += input.get_nth(i)
		}
		mean /= 4.0
		mut variance := 0.0
		for i in group_start .. group_start + 4 {
			difference := input.get_nth(i) - mean
			variance += difference * difference
		}
		inv_std := 1.0 / math.sqrt(variance / 4.0 + 1e-5)
		for i in group_start .. group_start + 4 {
			assert math.abs(output.get_nth(i) - (input.get_nth(i) - mean) * inv_std) < 1e-12
		}
	}
}

fn test_group_norm_backward_matches_finite_differences() ! {
	input_values := [1.0, 2.0, 4.0, 0.0]
	gamma_values := [1.5, 0.5]
	beta_values := [0.1, -0.2]
	gradient_values := [0.3, -0.7, 1.1, 1.2]
	input := vtl.from_array(input_values, [1, 2, 2])!
	gamma := vtl.from_array(gamma_values, [2])!
	beta := vtl.from_array(beta_values, [2])!
	gradient := vtl.from_array(gradient_values, [1, 2, 2])!
	grads := group_norm_backward[f64](gradient, input, 1, gamma, beta, 1e-5)!
	step := 1e-6
	for i in 0 .. input_values.len {
		mut plus := input_values.clone()
		mut minus := input_values.clone()
		plus[i] += step
		minus[i] -= step
		numeric := (group_norm_test_loss(plus, gamma_values, beta_values, gradient_values)! - group_norm_test_loss(minus,
			gamma_values, beta_values, gradient_values)!) / (2.0 * step)
		assert math.abs(grads[0].get_nth(i) - numeric) < 2e-6
	}
	for i in 0 .. gamma_values.len {
		mut plus := gamma_values.clone()
		mut minus := gamma_values.clone()
		plus[i] += step
		minus[i] -= step
		numeric := (group_norm_test_loss(input_values, plus, beta_values, gradient_values)! - group_norm_test_loss(input_values,
			minus, beta_values, gradient_values)!) / (2.0 * step)
		assert math.abs(grads[1].get_nth(i) - numeric) < 2e-6
	}
	for i in 0 .. beta_values.len {
		mut plus := beta_values.clone()
		mut minus := beta_values.clone()
		plus[i] += step
		minus[i] -= step
		numeric := (group_norm_test_loss(input_values, gamma_values, plus, gradient_values)! - group_norm_test_loss(input_values,
			gamma_values, minus, gradient_values)!) / (2.0 * step)
		assert math.abs(grads[2].get_nth(i) - numeric) < 2e-6
	}
}

fn test_group_norm_rejects_invalid_group_and_affine_shapes() ! {
	input := vtl.from_array([1.0, 2.0, 3.0, 4.0], [1, 2, 2])!
	wrong_gamma := vtl.ones[f64]([3])
	if _ := group_norm_forward[f64](input, 3, unsafe { nil }, unsafe { nil }, 1e-5) {
		assert false, 'expected non-divisible group count to fail'
	}
	if _ := group_norm_forward[f64](input, 1, wrong_gamma, unsafe { nil }, 1e-5) {
		assert false, 'expected mismatched gamma shape to fail'
	}
}
