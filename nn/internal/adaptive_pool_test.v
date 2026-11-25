module internal

import vtl

fn test_adaptive_avgpool2d_forward_uses_fixed_output_shape() ! {
	input := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1, 1, 3, 3])!
	result := adaptive_avgpool2d_forward[f64](input, [2, 2])!

	assert result.shape == [1, 1, 2, 2]
	assert result.get([0, 0, 0, 0]) == 3.0
	assert result.get([0, 0, 0, 1]) == 4.0
	assert result.get([0, 0, 1, 0]) == 6.0
	assert result.get([0, 0, 1, 1]) == 7.0
}

fn test_adaptive_avgpool2d_backward_accumulates_region_gradients() ! {
	input := vtl.ones[f64]([1, 1, 3, 3])
	grad_out := vtl.ones[f64]([1, 1, 2, 2])
	result := adaptive_avgpool2d_backward[f64](grad_out, input, [2, 2])!

	assert result.get([0, 0, 0, 0]) == 0.25
	assert result.get([0, 0, 1, 1]) == 1.0
	assert result.get([0, 0, 2, 2]) == 0.25
}

fn test_adaptive_avgpool2d_rejects_invalid_output_size() ! {
	input := vtl.ones[f64]([1, 1, 3, 3])
	if _ := adaptive_avgpool2d_forward[f64](input, [0, 2]) {
		assert false
	}
}
