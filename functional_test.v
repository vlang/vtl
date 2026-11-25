module vtl

import vtl.stats

fn test_vmap_maps_vectorized_samples_and_supports_negative_axes() ! {
	input := from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!
	result := vmap[f64](fn (sample &Tensor[f64]) !&Tensor[f64] {
		return sample.multiply_scalar(2.0)
	}, input, 0, -1)!
	assert result.shape == [2, 2]
	assert result.to_array() == [2.0, 6.0, 4.0, 8.0]
}

fn test_vmap_maps_scalar_outputs() ! {
	input := from_array([1.0, 2.0, 3.0], [3])!
	result := vmap[f64](fn (sample &Tensor[f64]) !&Tensor[f64] {
		return from_1d([stats.sum[f64](sample)])
	}, input, 0, 0)!
	assert result.shape == [3, 1]
	assert result.to_array() == [1.0, 2.0, 3.0]
}
