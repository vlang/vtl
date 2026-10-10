module internal

import math
import vtl

fn rms_norm_test_loss(input_values []f64, weight_values []f64, gradient_values []f64) !f64 {
	input := vtl.from_array(input_values, [2, 2])!
	weight := vtl.from_array(weight_values, [2])!
	gradient := vtl.from_array(gradient_values, [2, 2])!
	output := rms_norm_forward[f64](input, weight, [2], 1e-5)!
	mut loss := 0.0
	for i in 0 .. output.size {
		loss += output.get_nth(i) * gradient.get_nth(i)
	}
	return loss
}

fn test_rms_norm_forward_normalizes_trailing_blocks_with_weight() ! {
	input := vtl.from_array([3.0, 4.0, 5.0, 12.0], [2, 2])!
	weight := vtl.from_array([2.0, 0.5], [2])!
	output := rms_norm_forward[f64](input, weight, [2], 0.0)!
	assert math.abs(output.get_nth(0) - 3.0 / math.sqrt(12.5) * 2.0) < 1e-12
	assert math.abs(output.get_nth(1) - 4.0 / math.sqrt(12.5) * 0.5) < 1e-12
	assert math.abs(output.get_nth(2) - 5.0 / math.sqrt(84.5) * 2.0) < 1e-12
	assert math.abs(output.get_nth(3) - 12.0 / math.sqrt(84.5) * 0.5) < 1e-12
}

fn test_rms_norm_backward_matches_finite_differences_for_input_and_weight() ! {
	input_values := [1.0, 2.0, 4.0, 3.0]
	weight_values := [1.5, 0.5]
	gradient_values := [0.3, -0.7, 1.1, 1.2]
	input := vtl.from_array(input_values, [2, 2])!
	weight := vtl.from_array(weight_values, [2])!
	gradient := vtl.from_array(gradient_values, [2, 2])!
	grads := rms_norm_backward[f64](gradient, input, weight, [2], 1e-5)!
	step := 1e-6
	for i in 0 .. input_values.len {
		mut plus := input_values.clone()
		mut minus := input_values.clone()
		plus[i] += step
		minus[i] -= step
		numeric := (rms_norm_test_loss(plus, weight_values, gradient_values)! - rms_norm_test_loss(minus,
			weight_values, gradient_values)!) / (2.0 * step)
		assert math.abs(grads[0].get_nth(i) - numeric) < 2e-6
	}
	for i in 0 .. weight_values.len {
		mut plus := weight_values.clone()
		mut minus := weight_values.clone()
		plus[i] += step
		minus[i] -= step
		numeric := (rms_norm_test_loss(input_values, plus, gradient_values)! - rms_norm_test_loss(input_values,
			minus, gradient_values)!) / (2.0 * step)
		assert math.abs(grads[1].get_nth(i) - numeric) < 2e-6
	}
}

fn test_rms_norm_without_weight_and_validation() ! {
	input := vtl.from_array([3.0, 4.0, 5.0, 12.0], [2, 2])!
	output := rms_norm_forward[f32](vtl.from_array([f32(3), 4, 5, 12], [2, 2])!, unsafe { nil },
		[2], 0.0)!
	assert output.shape == input.shape
	assert math.abs(f64(output.get_nth(0)) - 0.8485281) < 1e-6
	if _ := rms_norm_forward[f64](input, unsafe { nil }, [3], 1e-5) {
		assert false, 'expected invalid normalized shape to fail'
	}
	wrong_weight := vtl.ones[f64]([3])
	if _ := rms_norm_forward[f64](input, wrong_weight, [2], 1e-5) {
		assert false, 'expected mismatched weight shape to fail'
	}
	if _ := rms_norm_forward[f64](input, unsafe { nil }, [2], math.nan()) {
		assert false, 'expected non-finite epsilon to fail'
	}
}
