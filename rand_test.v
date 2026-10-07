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

fn test_random_i32_values_stay_within_requested_range() {
	random_seed(42)
	values := random[i32](-20, 20, [32], TensorData{})
	for value in values.to_array() {
		assert value >= -20 && value < 20
	}
}

fn test_random_i16_and_u8_values_stay_within_requested_ranges() {
	random_seed(43)
	signed := random[i16](-200, 201, [64], TensorData{})
	for value in signed.to_array() {
		assert value >= -200 && value < 201
	}

	unsigned := random[u8](10, 200, [64], TensorData{})
	for value in unsigned.to_array() {
		assert value >= 10 && value < 200
	}
}
