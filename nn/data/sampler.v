module data

import rand

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
