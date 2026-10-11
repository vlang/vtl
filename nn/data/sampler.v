module data

import rand
import math

// Sampler produces sample indices for a dataset of the provided size.
pub interface Sampler {
	sample_indices(dataset_size int) ![]int
}

// SequentialSampler returns every dataset index in order.
pub struct SequentialSampler {}

// sample_indices returns [0, 1, ..., dataset_size-1].
pub fn (s &SequentialSampler) sample_indices(dataset_size int) ![]int {
	if dataset_size < 0 {
		return error('dataset_size must be non-negative')
	}
	return []int{len: dataset_size, init: index}
}

// RandomSampler returns every dataset index once in a random order.
pub struct RandomSampler {}

// sample_indices shuffles a full index range using Fisher-Yates.
pub fn (s &RandomSampler) sample_indices(dataset_size int) ![]int {
	if dataset_size < 0 {
		return error('dataset_size must be non-negative')
	}
	mut indices := []int{len: dataset_size, init: index}
	for i := indices.len - 1; i > 0; i-- {
		j := rand.intn(i + 1) or { 0 }
		indices[i], indices[j] = indices[j], indices[i]
	}
	return indices
}

// SubsetRandomSampler returns the configured subset in a random order.
pub struct SubsetRandomSampler {
pub:
	indices []int
}

// SubsetRandomSampler.new snapshots the requested subset indices.
pub fn SubsetRandomSampler.new(indices []int) &SubsetRandomSampler {
	return &SubsetRandomSampler{
		indices: indices.clone()
	}
}

// sample_indices shuffles the subset after checking dataset bounds.
pub fn (s &SubsetRandomSampler) sample_indices(dataset_size int) ![]int {
	if dataset_size < 0 {
		return error('dataset_size must be non-negative')
	}
	mut indices := s.indices.clone()
	for i, index_value in indices {
		if index_value < 0 || index_value >= dataset_size {
			return error('sampler index ${index_value} at position ${i} is out of bounds for dataset size ${dataset_size}')
		}
	}
	for i := indices.len - 1; i > 0; i-- {
		j := rand.intn(i + 1) or { 0 }
		indices[i], indices[j] = indices[j], indices[i]
	}
	return indices
}

// WeightedRandomSampler draws indices in proportion to non-negative weights.
pub struct WeightedRandomSampler {
	weights     []f64
	num_samples int
	replacement bool
}

// WeightedRandomSampler.new snapshots weights and validates the draw count.
pub fn WeightedRandomSampler.new(weights []f64, num_samples int, replacement bool) !&WeightedRandomSampler {
	if num_samples < 0 {
		return error('num_samples must be non-negative')
	}
	if weights.len == 0 {
		return error('weights must not be empty')
	}
	mut positive_count := 0
	for weight in weights {
		if weight < 0 || math.is_nan(weight) || math.is_inf(weight, 0) {
			return error('weights must be finite and non-negative')
		}
		if weight > 0 {
			positive_count++
		}
	}
	if num_samples > 0 && positive_count == 0 {
		return error('weights must contain a positive value when num_samples is positive')
	}
	if !replacement && num_samples > positive_count {
		return error('num_samples exceeds the number of positive weights without replacement')
	}
	return &WeightedRandomSampler{
		weights:     weights.clone()
		num_samples: num_samples
		replacement: replacement
	}
}

// sample_indices draws the configured number of weighted dataset indices.
pub fn (s &WeightedRandomSampler) sample_indices(dataset_size int) ![]int {
	if dataset_size < 0 {
		return error('dataset_size must be non-negative')
	}
	if s.weights.len != dataset_size {
		return error('weights length ${s.weights.len} must match dataset size ${dataset_size}')
	}
	if s.num_samples == 0 {
		return []int{}
	}
	mut scaled_weights := s.weights.clone()
	mut max_weight := 0.0
	for weight in scaled_weights {
		max_weight = math.max(max_weight, weight)
	}
	if max_weight <= 0 {
		return error('weights must contain a positive value when num_samples is positive')
	}
	for i in 0 .. scaled_weights.len {
		scaled_weights[i] /= max_weight
	}
	mut total_weight := 0.0
	for weight in scaled_weights {
		total_weight += weight
	}
	mut result := []int{cap: s.num_samples}
	for _ in 0 .. s.num_samples {
		target := rand.f64_in_range(0.0, total_weight)!
		mut cumulative := 0.0
		mut selected_index := -1
		for index, weight in scaled_weights {
			if weight <= 0 {
				continue
			}
			cumulative += weight
			if target < cumulative {
				selected_index = index
				break
			}
		}
		if selected_index < 0 {
			for index, weight in scaled_weights {
				if weight > 0 {
					selected_index = index
				}
			}
		}
		result << selected_index
		if !s.replacement {
			total_weight -= scaled_weights[selected_index]
			scaled_weights[selected_index] = 0.0
		}
	}
	return result
}
