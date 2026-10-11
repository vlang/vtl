module data

import vsl.la
import vsl.errors
import math

// Dataset is an interface for accessing data in batches
pub interface Dataset[T] {
	// len returns the number of samples in the dataset
	len() int
	// get returns the input and label at index i
	get(i int) !(&la.Matrix[T], []T)
}

// Subset wraps a Dataset and provides indices for a subset of the data

// Subset defines a public data structure for this module.

// Subset defines a public data structure for this module.
@[heap]
pub struct Subset[T] {
pub:
	dataset &Dataset[T]
	indices []int
}

// Subset.new creates a new Subset from a Dataset
pub fn Subset.new[T](dataset &Dataset[T], indices []int) &Subset[T] {
	return &Subset[T]{
		dataset: dataset
		indices: indices
	}
}

// len returns the number of samples in the subset
pub fn (s &Subset[T]) len() int {
	return s.indices.len
}

// get returns the input and label at subset index i
pub fn (s &Subset[T]) get(i int) !(&la.Matrix[T], []T) {
	if i < 0 || i >= s.len() {
		return errors.error('index ${i} out of bounds for subset of size ${s.len()}', .einval)
	}
	return s.dataset.get(s.indices[i])!
}

// split splits a Subset into train and validation subsets
// train_ratio is the proportion of data to use for training
pub fn (s &Subset[T]) split(train_ratio f64) !(&Subset[T], &Subset[T]) {
	if math.is_nan(train_ratio) || math.is_inf(train_ratio, 0) || train_ratio < 0.0
		|| train_ratio > 1.0 {
		return errors.error('split ratio must be finite and between 0 and 1', .einval)
	}
	n := s.len()
	n_train := int(f64(n) * train_ratio)
	if n_train == 0 || n_train == n {
		return errors.error('split ratio ${train_ratio} results in empty train or val set', .einval)
	}
	train_indices := s.indices[..n_train].clone()
	val_indices := s.indices[n_train..].clone()
	return Subset.new[T](s.dataset, train_indices), Subset.new[T](s.dataset, val_indices)
}

// DataLoader provides mini-batch iteration over data

// DataLoader defines a public data structure for this module.

// DataLoader defines a public data structure for this module.
@[heap]
pub struct DataLoader[T] {
pub:
	data       &la.Matrix[T] // full data matrix [nb_samples][nb_features]
	labels     []T           // labels [nb_samples]
	batch_size int
	shuffle    bool
	sampler    Sampler
pub mut:
	indices          []int // sample order for the current epoch
	position         int   // current position in epoch
	resample_on_next bool
}

// DataLoader.new creates a new DataLoader
// data is [nb_samples][nb_features] matrix
// labels is [nb_samples] vector
pub fn DataLoader.new[T](data &la.Matrix[T], labels []T, batch_size int, shuffle bool) &DataLoader[T] {
	n := data.m
	indices := []int{len: n, init: index}
	mut sampler := Sampler(&SequentialSampler{})
	if shuffle {
		sampler = &RandomSampler{}
	}
	return &DataLoader[T]{
		data:             data
		labels:           labels
		batch_size:       batch_size
		shuffle:          shuffle
		sampler:          sampler
		indices:          indices
		position:         0
		resample_on_next: shuffle
	}
}

// new_with_sampler creates a DataLoader that obtains its sample order from a
// custom sampler. The sampler is called once per epoch, including after reset.
pub fn DataLoader.new_with_sampler[T](data &la.Matrix[T], labels []T, batch_size int, sampler Sampler) !&DataLoader[T] {
	if batch_size <= 0 {
		return error('batch_size must be positive')
	}
	if labels.len != data.m {
		return error('labels length ${labels.len} must match dataset size ${data.m}')
	}
	indices := sampler.sample_indices(data.m)!
	validate_sample_indices(indices, data.m)!
	return &DataLoader[T]{
		data:       data
		labels:     labels
		batch_size: batch_size
		shuffle:    false
		sampler:    sampler
		indices:    indices
		position:   0
	}
}

fn validate_sample_indices(indices []int, dataset_size int) ! {
	for i, sample_index in indices {
		if sample_index < 0 || sample_index >= dataset_size {
			return error('sampler index ${sample_index} at position ${i} is out of bounds for dataset size ${dataset_size}')
		}
	}
}

// len returns the number of batches per epoch
pub fn (dl &DataLoader[T]) len() int {
	if dl.batch_size <= 0 {
		return 0
	}
	return int(math.ceil(f64(dl.indices.len) / f64(dl.batch_size)))
}

// next returns the next batch of data and labels
// returns an error when no more batches are available
pub fn (mut dl DataLoader[T]) next() !(&la.Matrix[T], []T) {
	if dl.batch_size <= 0 {
		return error('batch_size must be positive')
	}
	if dl.labels.len != dl.data.m {
		return error('labels length ${dl.labels.len} must match dataset size ${dl.data.m}')
	}
	if dl.position == 0 && dl.resample_on_next {
		dl.indices = dl.sampler.sample_indices(dl.data.m)!
		validate_sample_indices(dl.indices, dl.data.m)!
		dl.resample_on_next = false
	}
	// Check if we have more batches
	if dl.position >= dl.indices.len {
		return errors.error('no more batches available, call reset() to start a new epoch',
			.efailed)
	}

	// Calculate batch bounds
	start := dl.position
	end := math.min(dl.position + dl.batch_size, dl.indices.len)
	actual_batch_size := end - start

	// Get batch indices
	batch_indices := dl.indices[start..end]

	// Extract batch data
	mut batch_data := la.Matrix.new[T](actual_batch_size, dl.data.n)
	for i := 0; i < actual_batch_size; i++ {
		for j := 0; j < dl.data.n; j++ {
			batch_data.set(i, j, dl.data.get(batch_indices[i], j))
		}
	}

	// Extract batch labels
	mut batch_labels := []T{len: actual_batch_size}
	for i := 0; i < actual_batch_size; i++ {
		batch_labels[i] = dl.labels[batch_indices[i]]
	}

	// Update position
	dl.position = end

	return batch_data, batch_labels
}

// reset resets the DataLoader to start a new epoch
pub fn (mut dl DataLoader[T]) reset() {
	dl.position = 0
	dl.resample_on_next = true
}

// TensorDataset implements Dataset for matrix data

// TensorDataset defines a public data structure for this module.

// TensorDataset defines a public data structure for this module.
@[heap]
pub struct TensorDataset[T] {
pub mut:
	x &la.Matrix[T]
	y []T
}

// TensorDataset.new creates a new TensorDataset
pub fn TensorDataset.new[T](x &la.Matrix[T], y []T) &TensorDataset[T] {
	return &TensorDataset[T]{
		x: x
		y: y
	}
}

// len returns the number of samples
pub fn (ds &TensorDataset[T]) len() int {
	return ds.x.m
}

// get returns the input matrix and labels at index i
pub fn (ds &TensorDataset[T]) get(i int) !(&la.Matrix[T], []T) {
	if i < 0 || i >= ds.len() {
		return errors.error('index ${i} out of bounds', .einval)
	}
	mut x := la.Matrix.new[T](1, ds.x.n)
	for j in 0 .. ds.x.n {
		x.set(0, j, ds.x.get(i, j))
	}
	return x, [ds.y[i]]
}
