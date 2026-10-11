module layers

import vtl

fn test_linear_forward_adds_row_bias_for_batched_outputs() ! {
	input := vtl.from_array([f32(1), 2, 3, 4], [2, 2])!
	weights := vtl.from_array([f32(1), 0, 0, 1], [2, 2])!
	bias := vtl.from_array([f32(10), 20], [1, 2])!

	output := linear_forward_f32_cpu(input, weights, bias)!

	assert output.shape == [2, 2]
	assert output.to_array() == [f32(11), 22, 13, 24]
}

fn test_linear_forward_adds_row_bias_for_batched_sequence_outputs() ! {
	input := vtl.from_array([f64(1), 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	weights := vtl.from_array([f64(1), 0, 0, 1], [2, 2])!
	bias := vtl.from_array([f64(10), 20], [1, 2])!

	output := linear_forward_f64_cpu(input, weights, bias)!

	assert output.shape == [2, 2, 2]
	assert output.to_array() == [11.0, 22, 13, 24, 15, 26, 17, 28]
}

fn test_linear_add_bias_falls_back_to_general_broadcasting() ! {
	mut output := vtl.from_array([f64(1), 2, 3, 4], [2, 2])!
	bias := vtl.from_array([f64(10), 20], [2, 1])!

	result := linear_add_bias[f64](mut output, bias)!

	assert result.to_array() == [11.0, 12, 23, 24]
}

fn test_linear_add_bias_supports_integer_tensors() ! {
	mut output := vtl.from_array([i32(1), 2, 3, 4], [2, 2])!
	bias := vtl.from_array([i32(10), 20], [1, 2])!

	result := linear_add_bias[i32](mut output, bias)!

	assert result.to_array() == [i32(11), 22, 13, 24]
}
