module main

import vtl
import vtl.stats

fn main() {
	mut training_rng := vtl.new_random_generator(42)
	mut validation_rng := vtl.new_random_generator(2026)
	training_features := training_rng.normal([4, 3], vtl.NormalTensorData{ mu: 0.0, sigma: 1.0 })!
	validation_mask := validation_rng.bernoulli(0.75, [4])!
	waiting_times := validation_rng.geometric(0.1, [4])!
	training_indices := vtl.from_1d([0, 1, 2, 3, 4, 5, 6, 7])!
	batch_indices := training_rng.choice[int](training_indices, 4, false)!
	weights := vtl.from_1d([1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])!
	weighted_indices := training_rng.choice_weighted[int](training_indices, weights, 4, true)!
	class_probabilities := vtl.from_1d([0.6, 0.3, 0.1])!
	class_counts := training_rng.multinomial(12, class_probabilities, [4])!
	epoch_order := training_rng.permutation(8)!
	shuffled_features := training_rng.permutation_tensor[f64](training_features)!
	shuffled_feature_columns := training_rng.permutation_axis[f64](training_features, -1)!
	positive_noise := training_rng.gamma(2.0, 0.5, [4])!
	probability_samples := training_rng.beta(2.0, 5.0, [4])!
	class_concentration := vtl.from_1d([0.5, 1.5, 3.0])!
	class_probabilities_sample := training_rng.dirichlet[f64](class_concentration, [2])!
	positive_scales := training_rng.lognormal(0.0, 0.25, [4])!
	noise := training_rng.uniform(-0.01, 0.01, [4])!
	event_counts := training_rng.binomial(12, 0.25, [4])!
	arrival_counts := validation_rng.poisson(3.5, [4])!
	lifetimes := validation_rng.weibull(1.5, [4])!
	chi_square_samples := validation_rng.chi_square(5.0, [4])!
	test_statistics := validation_rng.student_t(7.0, [4])!
	variance_ratios := validation_rng.f_distribution(5.0, 10.0, [4])!
	waiting_durations := validation_rng.exponential(0.5, [4])!

	println('Training batch shape: ${training_features.shape}')
	println('Training feature mean: ${stats.mean(training_features)}')
	println('Validation mask: ${validation_mask.to_array()}')
	println('Validation waiting times: ${waiting_times.to_array()}')
	println('Sampled training indices: ${batch_indices.to_array()}')
	println('Weighted training indices: ${weighted_indices.to_array()}')
	println('Multinomial class counts: ${class_counts.to_array()}')
	println('Shuffled epoch order: ${epoch_order.to_array()}')
	println('Shuffled feature rows shape: ${shuffled_features.shape}')
	println('Shuffled feature columns shape: ${shuffled_feature_columns.shape}')
	println('Positive gamma samples: ${positive_noise.to_array()}')
	println('Beta probability samples: ${probability_samples.to_array()}')
	println('Dirichlet class probability samples: ${class_probabilities_sample.to_array()}')
	println('Log-normal positive scales: ${positive_scales.to_array()}')
	println('Binomial event counts: ${event_counts.to_array()}')
	println('Poisson arrival counts: ${arrival_counts.to_array()}')
	println('Weibull lifetimes: ${lifetimes.to_array()}')
	println('Chi-square samples: ${chi_square_samples.to_array()}')
	println('Student t samples: ${test_statistics.to_array()}')
	println('F-distribution ratios: ${variance_ratios.to_array()}')
	println('Exponential waiting durations: ${waiting_durations.to_array()}')
	println('Independent augmentation noise: ${noise.to_array()}')

	training_rng.free()
	validation_rng.free()
}
