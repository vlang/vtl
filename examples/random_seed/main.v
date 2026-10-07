module main

import vtl

fn main() {
	vtl.random_seed(42)
	first := vtl.random[f64](0.0, 1.0, [4], vtl.TensorData{})
	println('First sequence: ${first.to_array()}')

	vtl.random_seed(42)
	repeated := vtl.random[f64](0.0, 1.0, [4], vtl.TensorData{})
	println('Repeated sequence: ${repeated.to_array()}')
	assert first.array_equal(repeated)

	vtl.random_seed(2026)
	arrival_counts := vtl.poisson(3.5, [4], vtl.TensorData{})!
	lifetimes := vtl.weibull(1.5, [4], vtl.TensorData{})!
	vtl.random_seed(2026)
	assert arrival_counts.array_equal(vtl.poisson(3.5, [4], vtl.TensorData{})!)
	assert lifetimes.array_equal(vtl.weibull(1.5, [4], vtl.TensorData{})!)
	println('Poisson event counts: ${arrival_counts.to_array()}')
	println('Weibull lifetimes: ${lifetimes.to_array()}')

	vtl.random_seed(2027)
	gamma_values := vtl.gamma(2.0, 3.0, [4], vtl.TensorData{})!
	beta_values := vtl.beta(2.0, 5.0, [4], vtl.TensorData{})!
	chi_square_values := vtl.chi_square(4.0, [4], vtl.TensorData{})!
	student_t_values := vtl.student_t(12.0, [4], vtl.TensorData{})!
	f_values := vtl.f_distribution(5.0, 20.0, [4], vtl.TensorData{})!
	vtl.random_seed(2027)
	assert gamma_values.array_equal(vtl.gamma(2.0, 3.0, [4], vtl.TensorData{})!)
	assert beta_values.array_equal(vtl.beta(2.0, 5.0, [4], vtl.TensorData{})!)
	assert chi_square_values.array_equal(vtl.chi_square(4.0, [4], vtl.TensorData{})!)
	assert student_t_values.array_equal(vtl.student_t(12.0, [4], vtl.TensorData{})!)
	assert f_values.array_equal(vtl.f_distribution(5.0, 20.0, [4], vtl.TensorData{})!)
	println('Gamma samples: ${gamma_values.to_array()}')
	println('Beta samples: ${beta_values.to_array()}')
	println('Chi-square samples: ${chi_square_values.to_array()}')
	println('Student t samples: ${student_t_values.to_array()}')
	println('F samples: ${f_values.to_array()}')
}
