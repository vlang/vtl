module vtl

fn test_random_seed_repeats_random_tensor_values() {
	random_seed(42)
	first := random[f64](0.0, 1.0, [16], TensorData{})
	random_seed(42)
	second := random[f64](0.0, 1.0, [16], TensorData{})
	assert first.array_equal(second)

	random_seed(43)
	different_seed := random[f64](0.0, 1.0, [16], TensorData{})
	assert !first.array_equal(different_seed)
}
