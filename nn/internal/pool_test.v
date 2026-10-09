module internal

import vtl

fn test_avgpool2d_backward_distributes_overlapping_window_gradients() ! {
	input := vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [1, 1, 3, 3])!
	grad_out := vtl.ones[f64]([1, 1, 2, 2])

	result := avgpool2d_backward[f64](grad_out, input, [2, 2], [0, 0], [1, 1])!

	expected := vtl.from_array([0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25],
		[1, 1, 3, 3])!
	assert result.array_equal(expected)
}

fn test_avgpool2d_backward_uses_full_window_area_with_padding() ! {
	input := vtl.ones[f64]([1, 1, 2, 2])
	grad_out := vtl.ones[f64]([1, 1, 3, 3])

	result := avgpool2d_backward[f64](grad_out, input, [2, 2], [1, 1], [1, 1])!

	assert result.to_array() == [1.0, 1.0, 1.0, 1.0]
}

fn test_avgpool2d_backward_preserves_batch_and_channel_gradients() ! {
	input := vtl.ones[f64]([2, 2, 2, 2])
	grad_out := vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2, 1, 1])!

	result := avgpool2d_backward[f64](grad_out, input, [2, 2], [0, 0], [2, 2])!

	assert result.to_array() == [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75,
		0.75, 1.0, 1.0, 1.0, 1.0]
}

fn test_avgpool2d_backward_rejects_invalid_gradient_shape() {
	input := vtl.ones[f64]([1, 1, 3, 3])
	grad_out := vtl.ones[f64]([1, 1, 1, 1])
	if _ := avgpool2d_backward[f64](grad_out, input, [2, 2], [0, 0], [1, 1]) {
		assert false, 'expected a gradient shape mismatch error'
	}
}
