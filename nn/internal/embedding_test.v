module internal

import vtl

fn test_embedding_backward_accumulates_repeated_indices() ! {
	input := vtl.from_array([1.0, 0.0, 1.0], [1, 3])!
	gradient := vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 3, 2])!
	weight := vtl.zeros[f64]([3, 2])

	result := embedding_backward[f64](gradient, input, weight)!
	assert result[0].to_array() == [3.0, 4.0, 6.0, 8.0, 0.0, 0.0]
}

fn test_embedding_backward_supports_f32() ! {
	input := vtl.from_array([f32(1.0), 1.0], [1, 2])!
	gradient := vtl.from_array([f32(1.0), 2.0, 3.0, 4.0], [1, 2, 2])!
	weight := vtl.zeros[f32]([2, 2])

	result := embedding_backward[f32](gradient, input, weight)!
	assert result[0].to_array() == [f32(0.0), 0.0, 4.0, 6.0]
}

fn test_embedding_backward_supports_strided_inputs() ! {
	input_base := vtl.from_array([0.0, 1.0, 2.0, 3.0], [2, 2])!
	input := input_base.transpose([1, 0])!
	gradient_base := vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])!
	gradient := gradient_base.transpose([1, 0, 2])!
	weight := vtl.zeros[f64]([4, 2])

	result := embedding_backward[f64](gradient, input, weight)!
	assert result[0].to_array() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
}

fn test_embedding_backward_rejects_incompatible_shapes() {
	input := vtl.zeros[f64]([2, 3])
	gradient := vtl.zeros[f64]([2, 3, 4])
	weight := vtl.zeros[f64]([5, 2])
	if _ := embedding_backward[f64](gradient, input, weight) {
		assert false, 'expected a shape mismatch error'
	}
}
