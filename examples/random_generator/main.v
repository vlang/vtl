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
	epoch_order := training_rng.permutation(8)!
	positive_noise := training_rng.gamma(2.0, 0.5, [4])!
	probability_samples := training_rng.beta(2.0, 5.0, [4])!
	positive_scales := training_rng.lognormal(0.0, 0.25, [4])!
	noise := training_rng.uniform(-0.01, 0.01, [4])!
	event_counts := training_rng.binomial(12, 0.25, [4])!
	waiting_durations := validation_rng.exponential(0.5, [4])!

	println('Training batch shape: ${training_features.shape}')
	println('Training feature mean: ${stats.mean(training_features)}')
	println('Validation mask: ${validation_mask.to_array()}')
	println('Validation waiting times: ${waiting_times.to_array()}')
	println('Sampled training indices: ${batch_indices.to_array()}')
	println('Shuffled epoch order: ${epoch_order.to_array()}')
	println('Positive gamma samples: ${positive_noise.to_array()}')
	println('Beta probability samples: ${probability_samples.to_array()}')
	println('Log-normal positive scales: ${positive_scales.to_array()}')
	println('Binomial event counts: ${event_counts.to_array()}')
	println('Exponential waiting durations: ${waiting_durations.to_array()}')
	println('Independent augmentation noise: ${noise.to_array()}')

	training_rng.free()
	validation_rng.free()
}
