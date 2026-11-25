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

fn test_adaptive_maxpool2d_forward_and_backward() ! {
	input := vtl.from_array[f64]([1.0, 5.0, 3.0, 4.0, 2.0, 6.0, 7.0, 8.0, 0.0], [1, 1, 3, 3])!
	result := adaptive_maxpool2d_forward[f64](input, [2, 2])!
	assert result.to_array() == [5.0, 6.0, 8.0, 8.0]
	grad_out := vtl.ones[f64]([1, 1, 2, 2])
	gradient := adaptive_maxpool2d_backward[f64](grad_out, input, [2, 2])!
	assert gradient.to_array() == [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0]
}

fn test_adaptive_avgpool1d_forward_and_backward() ! {
	input := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 5])!
	result := adaptive_avgpool1d_forward[f64](input, 2)!
	assert result.to_array() == [2.0, 4.0]
	grad_out := vtl.ones[f64]([1, 1, 2])
	gradient := adaptive_avgpool1d_backward[f64](grad_out, input, 2)!
	assert gradient.to_array() == [0.3333333333333333, 0.3333333333333333, 0.6666666666666666,
		0.3333333333333333, 0.3333333333333333]
}

fn test_adaptive_maxpool1d_forward_and_backward() ! {
	input := vtl.from_array[f64]([1.0, 5.0, 3.0, 4.0, 6.0], [1, 1, 5])!
	result := adaptive_maxpool1d_forward[f64](input, 2)!
	assert result.to_array() == [5.0, 6.0]
	grad_out := vtl.ones[f64]([1, 1, 2])
	gradient := adaptive_maxpool1d_backward[f64](grad_out, input, 2)!
	assert gradient.to_array() == [0.0, 1.0, 0.0, 0.0, 1.0]
}

fn test_adaptive_avgpool3d_forward_and_backward() ! {
	input := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 1, 2, 2, 2])!
	result := adaptive_avgpool3d_forward[f64](input, [1, 1, 1])!
	assert result.to_array() == [4.5]
	grad_out := vtl.ones[f64]([1, 1, 1, 1, 1])
	gradient := adaptive_avgpool3d_backward[f64](grad_out, input, [1, 1, 1])!
	assert gradient.to_array() == [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
}

fn test_adaptive_maxpool3d_forward_and_backward() ! {
	input := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 1, 2, 2, 2])!
	result := adaptive_maxpool3d_forward[f64](input, [1, 1, 1])!
	assert result.to_array() == [8.0]
	grad_out := vtl.ones[f64]([1, 1, 1, 1, 1])
	gradient := adaptive_maxpool3d_backward[f64](grad_out, input, [1, 1, 1])!
	assert gradient.to_array() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
}

fn test_avgpool1d_forward_and_backward() ! {
	input := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [1, 1, 4])!
	result := avgpool1d_forward[f64](input, 2, 0, 2)!
	assert result.to_array() == [1.5, 3.5]
	grad_out := vtl.ones[f64]([1, 1, 2])
	gradient := avgpool1d_backward[f64](grad_out, input, 2, 0, 2)!
	assert gradient.to_array() == [0.5, 0.5, 0.5, 0.5]
}

fn test_maxpool1d_forward_and_backward() ! {
	input := vtl.from_array[f64]([1.0, 3.0, 2.0, 4.0], [1, 1, 4])!
	result := maxpool1d_forward[f64](input, 2, 0, 2)!
	assert result.to_array() == [3.0, 4.0]
	grad_out := vtl.ones[f64]([1, 1, 2])
	gradient := maxpool1d_backward[f64](grad_out, input, 2, 0, 2)!
	assert gradient.to_array() == [0.0, 1.0, 0.0, 1.0]
}

fn test_pool1d_rejects_invalid_parameters() {
	input := vtl.ones[f64]([1, 1, 4])
	if _ := avgpool1d_forward[f64](input, 0, 0, 1) {
		assert false
	}
	if _ := maxpool1d_forward[f64](input, 2, 0, 0) {
		assert false
	}
	if _ := avgpool1d_forward[f64](input, 5, 0, 1) {
		assert false
	}
}

fn test_maxpool1d_ties_route_gradient_to_first_maximum() ! {
	input := vtl.from_array[f64]([2.0, 2.0, 1.0, 0.0], [1, 1, 4])!
	grad_out := vtl.ones[f64]([1, 1, 2])
	gradient := maxpool1d_backward[f64](grad_out, input, 2, 0, 2)!
	assert gradient.to_array() == [1.0, 0.0, 1.0, 0.0]
}
