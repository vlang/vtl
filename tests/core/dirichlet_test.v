module core

import vtl

fn test_seeded_dirichlet_samples_are_reproducible_and_normalized() ! {
	alpha := vtl.from_1d([0.5, 1.5, 3.0])!
	mut first_rng := vtl.new_random_generator(71)
	mut second_rng := vtl.new_random_generator(71)
	first := first_rng.dirichlet[f64](alpha, [4])!
	second := second_rng.dirichlet[f64](alpha, [4])!
	assert first.shape == [4, 3]
	assert first.to_array() == second.to_array()
	for row in 0 .. 4 {
		mut total := 0.0
		for column in 0 .. 3 {
			value := first.get([row, column])
			assert value > 0
			assert value < 1
			total += value
		}
		assert total > 0.999999999999
		assert total < 1.000000000001
	}
	first_rng.free()
	second_rng.free()
}

fn test_dirichlet_rejects_invalid_concentrations() {
	mut rng := vtl.new_random_generator(19)
	invalid_alpha := vtl.from_1d([1.0, 0.0])!
	_ := rng.dirichlet[f64](invalid_alpha, [1]) or {
		assert err.msg().contains('finite and positive')
		rng.free()
		return
	}
	assert false, 'Dirichlet must reject non-positive concentrations'
}

fn test_dirichlet_rejects_empty_concentrations() {
	mut rng := vtl.new_random_generator(23)
	empty_alpha := vtl.from_1d([]f64{})!
	_ := rng.dirichlet[f64](empty_alpha, [1]) or {
		assert err.msg().contains('non-empty vector')
		rng.free()
		return
	}
	assert false, 'Dirichlet must reject empty concentration vectors'
}
