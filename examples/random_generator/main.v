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
	noise := training_rng.uniform(-0.01, 0.01, [4])!

	println('Training batch shape: ${training_features.shape}')
	println('Training feature mean: ${stats.mean(training_features)}')
	println('Validation mask: ${validation_mask.to_array()}')
	println('Validation waiting times: ${waiting_times.to_array()}')
	println('Sampled training indices: ${batch_indices.to_array()}')
	println('Independent augmentation noise: ${noise.to_array()}')

	training_rng.free()
	validation_rng.free()
}
