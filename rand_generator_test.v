module vtl

fn test_random_generators_have_independent_reproducible_state() ! {
	mut first := new_random_generator(42)
	mut second := new_random_generator(7)
	first_values := first.uniform(0.0, 1.0, [8])!
	_ := second.uniform(0.0, 1.0, [32])!
	mut replay := new_random_generator(42)
	replayed_values := replay.uniform(0.0, 1.0, [8])!
	assert first_values.array_equal(replayed_values)
	first.free()
	second.free()
	replay.free()
}

fn test_random_generator_supports_normal_and_bernoulli_distributions() ! {
	mut generator := new_random_generator(123)
	normal_values := generator.normal([4], NormalTensorData{ mu: 5.0, sigma: 1.0 })!
	assert normal_values.shape == [4]
	for i in 0 .. normal_values.size {
		assert normal_values.get_nth(i) == normal_values.get_nth(i)
	}
	true_values := generator.bernoulli(1.0, [3])!
	false_values := generator.bernoulli(0.0, [3])!
	for i in 0 .. true_values.size {
		assert true_values.get_nth(i)
	}
	for i in 0 .. false_values.size {
		assert !false_values.get_nth(i)
	}
	generator.free()
}

fn test_random_generator_rejects_invalid_distribution_parameters() {
	mut generator := new_random_generator(1)
	if _ := generator.uniform(1.0, 0.0, [2]) {
		assert false, 'uniform must reject a reversed range'
	}
	if _ := generator.normal([2], NormalTensorData{ sigma: -1.0 }) {
		assert false, 'normal must reject a negative standard deviation'
	}
	if _ := generator.bernoulli(1.1, [2]) {
		assert false, 'bernoulli must reject probabilities above one'
	}
	generator.free()
}
