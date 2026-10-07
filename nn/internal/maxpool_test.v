module internal

import vtl

fn test_maxpool2d_forward_handles_negative_values_and_batch_offsets() {
	input := vtl.from_array([1.0, 2.0, 3.0, 4.0, 7.0, 6.0, 5.0, 8.0], [2, 1, 2, 2]) or {
		panic(err)
	}
	indices, output := maxpool2d[f64](input, [2, 2], [0, 0], [1, 1])
	assert output.shape == [2, 1, 1, 1]
	assert output.to_array() == [4.0, 8.0]
	assert indices.to_array() == [3, 7]

	channels := vtl.from_array([1.0, 2.0, 3.0, 4.0, 7.0, 6.0, 5.0, 8.0], [1, 2, 2, 2]) or {
		panic(err)
	}
	channel_indices, channel_output := maxpool2d[f64](channels, [2, 2], [0, 0], [1, 1])
	assert channel_output.to_array() == [4.0, 8.0]
	assert channel_indices.to_array() == [3, 7]

	negative := vtl.from_array([-5.0, -2.0, -3.0, -4.0], [1, 1, 2, 2]) or { panic(err) }
	negative_indices, negative_output := maxpool2d[f64](negative, [2, 2], [0, 0], [1, 1])
	assert negative_output.to_array() == [-2.0]
	assert negative_indices.to_array() == [1]
}

fn test_maxpool2d_backward_accumulates_overlapping_windows_and_uses_batch_offsets() ! {
	indices := vtl.from_array([1, 1, 7], [1, 1, 1, 3])!
	gradient := vtl.from_array([2.0, 3.0, 4.0], [1, 1, 1, 3])!
	result := maxpool2d_backward[f64]([2, 1, 2, 2], indices, gradient)!
	assert result.to_array() == [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]
}

fn test_maxpool2d_backward_rejects_invalid_saved_indices() {
	indices := vtl.from_array([-1], [1]) or { panic(err) }
	gradient := vtl.from_array([1.0], [1]) or { panic(err) }
	if _ := maxpool2d_backward[f64]([1, 1], indices, gradient) {
		assert false, 'expected an out-of-bounds saved index error'
	}
}
