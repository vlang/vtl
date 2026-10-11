module datasets

import vtl
import vtl.nn.data as nn_data
import math

// DataLoader provides an iterable over a dataset with batching and shuffling.
// Contiguous batches are zero-copy slice views. Batches whose indices are not
// contiguous (usually shuffled batches) are gathered into a new tensor.
//
// Design principles (inspired by Arraymancer and PyTorch):
// - Zero-copy for contiguous sample indices; shuffled batches may require copies
// - Generic: works with any tensor element type T (f64, f32)
// - Configurable: batch_size, shuffle, drop_last
// - Deterministic: optional seed for reproducible shuffling
// - Supports optional labels tensor for supervised learning

// DataLoader defines a public data structure for this module.

// DataLoader defines a public data structure for this module.
@[heap]
pub struct DataLoader[T] {
pub:
	dataset    &vtl.Tensor[T]
	labels     &vtl.Tensor[T] = unsafe { nil }
	batch_size int
	shuffle    bool
	drop_last  bool
	seed       u64
pub mut:
	indices []int
	epoch   int
mut:
	sampler            nn_data.Sampler
	uses_sampler       bool
	batch_sampler      nn_data.BatchSampler
	uses_batch_sampler bool
	batches            [][]int
	collate_fn         CollateFn[T] = unsafe { nil }
}

// DataLoaderConfig holds configuration for creating a DataLoader.
pub struct DataLoaderConfig {
pub:
	batch_size int  = 32
	shuffle    bool = true
	drop_last  bool = true
	seed       u64
}

// CollatedBatch contains the tensors produced for one selected batch. Labels
// are nil when the source loader has no labels or the collator omits them.
pub struct CollatedBatch[T] {
pub:
	features &vtl.Tensor[T]
	labels   &vtl.Tensor[T] = unsafe { nil }
}

// CollateFn builds tensors for the selected sample indices. `labels` is nil
// for feature-only loaders. The callback owns the collation policy, including
// output shapes and ordering.
pub type CollateFn[T] = fn (dataset &vtl.Tensor[T], labels &vtl.Tensor[T], indices []int) !CollatedBatch[T]

// batch returns a batch at index `i`. Contiguous indices return a view into the
// dataset; other index orders return a copy. Mutating a returned view also
// mutates the corresponding dataset elements. Returns none for invalid indices
// or a non-positive batch size.
pub fn (dl &DataLoader[T]) batch(i int) ?&vtl.Tensor[T] {
	batch_indices := dl.batch_indices(i) or { return none }
	if range := contiguous_range(batch_indices) {
		return dl.dataset.slice_hilo([range[0]], [range[1] + 1]) or { return none }
	}

	mut result_shape := dl.dataset.shape.clone()
	result_shape[0] = batch_indices.len

	mut result := vtl.zeros[T](result_shape)
	row_size := dl.dataset.size / dl.dataset.shape[0]
	for j, idx in batch_indices {
		src_base := idx * row_size
		dst_base := j * row_size
		for k := 0; k < row_size; k++ {
			result.set_nth(dst_base + k, dl.dataset.get_nth(src_base + k))
		}
	}
	return result
}

// batch_with_labels returns both features and labels for batch `i`.
// Both tensors are extracted using the same index set.
// Returns none if the index is out of range or if no labels tensor was provided.
pub fn (dl &DataLoader[T]) batch_with_labels(i int) ?(&vtl.Tensor[T], &vtl.Tensor[T]) {
	if dl.labels == unsafe { nil } {
		return none
	}
	batch_indices := dl.batch_indices(i) or { return none }
	if range := contiguous_range(batch_indices) {
		features := dl.dataset.slice_hilo([range[0]], [range[1] + 1]) or { return none }
		labels := dl.labels.slice_hilo([range[0]], [range[1] + 1]) or { return none }
		return features, labels
	}

	// Build features batch
	mut feat_shape := dl.dataset.shape.clone()
	feat_shape[0] = batch_indices.len
	mut feat_result := vtl.zeros[T](feat_shape)
	feat_row_size := dl.dataset.size / dl.dataset.shape[0]
	for j, idx in batch_indices {
		src_base := idx * feat_row_size
		dst_base := j * feat_row_size
		for k := 0; k < feat_row_size; k++ {
			feat_result.set_nth(dst_base + k, dl.dataset.get_nth(src_base + k))
		}
	}

	// Build labels batch (labels are 2D: [N, num_classes])
	label_row_size := dl.labels.size / dl.labels.shape[0]
	mut label_shape := dl.labels.shape.clone()
	label_shape[0] = batch_indices.len
	mut label_result := vtl.zeros[T](label_shape)
	for j, idx in batch_indices {
		src_base := idx * label_row_size
		dst_base := j * label_row_size
		for k := 0; k < label_row_size; k++ {
			label_result.set_nth(dst_base + k, dl.labels.get_nth(src_base + k))
		}
	}

	return feat_result, label_result
}

// set_collate_fn installs a callback used by collated_batch. Existing batch
// accessors keep their default tensor slicing and gathering behavior.
pub fn (mut dl DataLoader[T]) set_collate_fn(collate_fn CollateFn[T]) {
	dl.collate_fn = collate_fn
}

// collated_batch builds the batch selected by `i`. A custom callback receives
// the source tensors and a copy of the selected indices, so it can implement
// custom padding, filtering, transforms, or ordering without changing loader
// state. Without a callback, this returns the same default tensors as batch
// and batch_with_labels.
pub fn (dl &DataLoader[T]) collated_batch(i int) !CollatedBatch[T] {
	batch_indices := dl.batch_indices(i) or {
		return error('batch index ${i} is out of range')
	}
	if dl.collate_fn != unsafe { nil } {
		return dl.collate_fn(dl.dataset, dl.labels, batch_indices.clone())
	}
	if dl.labels == unsafe { nil } {
		features := dl.batch(i) or { return error('unable to build batch ${i}') }
		return CollatedBatch[T]{
			features: features
		}
	}
	features, labels := dl.batch_with_labels(i) or {
		return error('unable to build labeled batch ${i}')
	}
	return CollatedBatch[T]{
		features: features
		labels:   labels
	}
}

// len returns the number of batches (drop_last affects count).
pub fn (dl &DataLoader[T]) len() int {
	if dl.uses_batch_sampler {
		return dl.batches.len
	}
	if dl.indices.len == 0 || dl.batch_size <= 0 {
		return 0
	}
	if dl.drop_last {
		return dl.indices.len / dl.batch_size
	}
	return 1 + (dl.indices.len - 1) / dl.batch_size
}

fn (dl &DataLoader[T]) batch_indices(i int) ?[]int {
	if i < 0 || i >= dl.len() {
		return none
	}
	if dl.uses_batch_sampler {
		return dl.batches[i]
	}
	start := i * dl.batch_size
	if start >= dl.indices.len {
		return none
	}
	end := if dl.batch_size >= dl.indices.len - start {
		dl.indices.len
	} else {
		start + dl.batch_size
	}
	return dl.indices[start..end]
}

fn contiguous_range(indices []int) ?[]int {
	if indices.len == 0 {
		return none
	}
	for i in 1 .. indices.len {
		if indices[i] != indices[i - 1] + 1 {
			return none
		}
	}
	return [indices[0], indices[indices.len - 1]]
}

// total_samples returns the number of samples in the dataset.
pub fn (dl &DataLoader[T]) total_samples() int {
	return dl.indices.len
}

// reset re-shuffles the indices (call between epochs). Uses seed if set.
pub fn (mut dl DataLoader[T]) reset() {
	dl.reset_checked() or { panic('DataLoader.reset failed: ${err.msg()}') }
}

// reset_checked advances the loader to a new epoch and reports sampler errors.
pub fn (mut dl DataLoader[T]) reset_checked() ! {
	next_epoch := dl.epoch + 1
	if dl.uses_batch_sampler {
		batches := dl.batch_sampler.sample_batches(dl.dataset.shape[0])!
		validate_sampler_batches(batches, dl.dataset.shape[0])!
		dl.batches = batches
		dl.indices = flatten_sampler_batches(batches)
	} else if dl.uses_sampler {
		dl.indices = dl.sampler.sample_indices(dl.dataset.shape[0])!
		validate_sampler_indices(dl.indices, dl.dataset.shape[0])!
	} else if dl.shuffle {
		dl.epoch = next_epoch
		dl.shuffle_indices()
	}
	dl.epoch = next_epoch
}

// new_data_loader_with_batch_sampler_checked creates a loader using explicit
// sample batches from batch_sampler.
pub fn new_data_loader_with_batch_sampler_checked[T](dataset &vtl.Tensor[T], config DataLoaderConfig, batch_sampler nn_data.BatchSampler) !&DataLoader[T] {
	if dataset.shape.len == 0 {
		return error('dataset must have a sample dimension')
	}
	n := dataset.shape[0]
	batches := batch_sampler.sample_batches(n)!
	validate_sampler_batches(batches, n)!
	return &DataLoader[T]{
		dataset:            dataset
		batch_size:         config.batch_size
		shuffle:            false
		drop_last:          false
		seed:               config.seed
		sampler:            &nn_data.SequentialSampler{}
		batch_sampler:      batch_sampler
		uses_batch_sampler: true
		batches:            batches
		indices:            flatten_sampler_batches(batches)
	}
}

// shuffle_indices shuffles the index array in-place using Fisher-Yates.
fn (mut dl DataLoader[T]) shuffle_indices() {
	mut rng := rng_from_seed(dl.seed + u64(dl.epoch))
	mut i := dl.indices.len - 1
	for i > 0 {
		j := int(rng.next() % u64(i + 1))
		dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		i--
	}
}

// new_data_loader creates a DataLoader from a dataset tensor (features only).
pub fn new_data_loader[T](dataset &vtl.Tensor[T], config DataLoaderConfig) &DataLoader[T] {
	n := if dataset.shape.len > 0 { dataset.shape[0] } else { 0 }
	mut indices := []int{len: n}
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	mut dl := &DataLoader[T]{
		dataset:       dataset
		batch_size:    config.batch_size
		shuffle:       config.shuffle
		drop_last:     config.drop_last
		seed:          config.seed
		sampler:       &nn_data.SequentialSampler{}
		batch_sampler: &nn_data.FixedBatchSampler{}
		indices:       indices
		epoch:         0
	}
	if config.shuffle {
		dl.shuffle_indices()
	}
	return dl
}

// new_data_loader_with_sampler_checked creates a loader whose sample order is
// supplied by sampler. The sampler is invoked again by reset_checked each epoch.
pub fn new_data_loader_with_sampler_checked[T](dataset &vtl.Tensor[T], config DataLoaderConfig, sampler nn_data.Sampler) !&DataLoader[T] {
	if dataset.shape.len == 0 {
		return error('dataset must have a sample dimension')
	}
	n := dataset.shape[0]
	indices := sampler.sample_indices(n)!
	validate_sampler_indices(indices, n)!
	return &DataLoader[T]{
		dataset:       dataset
		batch_size:    config.batch_size
		shuffle:       false
		drop_last:     config.drop_last
		seed:          config.seed
		sampler:       sampler
		batch_sampler: &nn_data.FixedBatchSampler{}
		uses_sampler:  true
		indices:       indices
	}
}

// new_data_loader_with_labels creates a DataLoader with both features and labels tensors.
// Both tensors must have the same first dimension (number of samples). This
// compatibility constructor panics for invalid shapes; prefer the checked
// variant when input dimensions are not already trusted.
pub fn new_data_loader_with_labels[T](dataset &vtl.Tensor[T], labels &vtl.Tensor[T], config DataLoaderConfig) &DataLoader[T] {
	return new_data_loader_with_labels_checked[T](dataset, labels, config) or {
		panic(err.msg())
	}
}

// new_data_loader_with_labels_checked creates a labeled DataLoader and reports
// invalid dataset or label shapes as an error instead of deferring failure
// until a batch is read.
pub fn new_data_loader_with_labels_checked[T](dataset &vtl.Tensor[T], labels &vtl.Tensor[T], config DataLoaderConfig) !&DataLoader[T] {
	if dataset.shape.len == 0 {
		return error('dataset must have a sample dimension')
	}
	if labels.shape.len == 0 {
		return error('labels must have a sample dimension')
	}
	n := dataset.shape[0]
	if labels.shape[0] != n {
		return error('dataset and labels must have the same number of samples: ${n} != ${labels.shape[0]}')
	}
	mut indices := []int{len: n}
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	mut dl := &DataLoader[T]{
		dataset:       dataset
		labels:        labels
		batch_size:    config.batch_size
		shuffle:       config.shuffle
		drop_last:     config.drop_last
		seed:          config.seed
		sampler:       &nn_data.SequentialSampler{}
		batch_sampler: &nn_data.FixedBatchSampler{}
		indices:       indices
		epoch:         0
	}
	if config.shuffle {
		dl.shuffle_indices()
	}
	return dl
}

// new_data_loader_with_labels_and_sampler_checked creates a labeled loader
// with a custom sample order and validates the data and sampler indices.
pub fn new_data_loader_with_labels_and_sampler_checked[T](dataset &vtl.Tensor[T], labels &vtl.Tensor[T], config DataLoaderConfig, sampler nn_data.Sampler) !&DataLoader[T] {
	if dataset.shape.len == 0 {
		return error('dataset must have a sample dimension')
	}
	if labels.shape.len == 0 {
		return error('labels must have a sample dimension')
	}
	n := dataset.shape[0]
	if labels.shape[0] != n {
		return error('dataset and labels must have the same number of samples: ${n} != ${labels.shape[0]}')
	}
	indices := sampler.sample_indices(n)!
	validate_sampler_indices(indices, n)!
	return &DataLoader[T]{
		dataset:       dataset
		labels:        labels
		batch_size:    config.batch_size
		shuffle:       false
		drop_last:     config.drop_last
		seed:          config.seed
		sampler:       sampler
		batch_sampler: &nn_data.FixedBatchSampler{}
		uses_sampler:  true
		indices:       indices
	}
}

// new_data_loader_with_labels_and_batch_sampler_checked creates a labeled
// loader using explicit sample batches and preserves feature-label alignment.
pub fn new_data_loader_with_labels_and_batch_sampler_checked[T](dataset &vtl.Tensor[T], labels &vtl.Tensor[T], config DataLoaderConfig, batch_sampler nn_data.BatchSampler) !&DataLoader[T] {
	if dataset.shape.len == 0 {
		return error('dataset must have a sample dimension')
	}
	if labels.shape.len == 0 {
		return error('labels must have a sample dimension')
	}
	n := dataset.shape[0]
	if labels.shape[0] != n {
		return error('dataset and labels must have the same number of samples: ${n} != ${labels.shape[0]}')
	}
	batches := batch_sampler.sample_batches(n)!
	validate_sampler_batches(batches, n)!
	return &DataLoader[T]{
		dataset:            dataset
		labels:             labels
		batch_size:         config.batch_size
		shuffle:            false
		drop_last:          false
		seed:               config.seed
		sampler:            &nn_data.SequentialSampler{}
		batch_sampler:      batch_sampler
		uses_batch_sampler: true
		batches:            batches
		indices:            flatten_sampler_batches(batches)
	}
}

fn validate_sampler_indices(indices []int, dataset_size int) ! {
	for i, sample_index in indices {
		if sample_index < 0 || sample_index >= dataset_size {
			return error('sampler index ${sample_index} at position ${i} is out of bounds for dataset size ${dataset_size}')
		}
	}
}

fn validate_sampler_batches(batches [][]int, dataset_size int) ! {
	for batch_index, batch in batches {
		if batch.len == 0 {
			return error('batch sampler returned an empty batch at position ${batch_index}')
		}
		validate_sampler_indices(batch, dataset_size)!
	}
}

fn flatten_sampler_batches(batches [][]int) []int {
	mut indices := []int{}
	for batch in batches {
		indices << batch
	}
	return indices
}

// DataLoaderIterator provides a for-in loop interface over a DataLoader.
pub struct DataLoaderIterator[T] {
mut:
	loader  &DataLoader[T]
	current int
}

// iter returns an iterator over the DataLoader batches.
pub fn (dl &DataLoader[T]) iter[T]() DataLoaderIterator[T] {
	return DataLoaderIterator[T]{
		loader:  dl
		current: 0
	}
}

// next returns the next batch. Returns none when exhausted.
pub fn (mut it DataLoaderIterator[T]) next[T]() ?&vtl.Tensor[T] {
	if it.current >= it.loader.len() {
		return none
	}
	defer {
		it.current++
	}
	return it.loader.batch(it.current) or { none }
}

// for_each applies `fn` to each batch until exhaustion or `fn` returns false.
pub fn (dl &DataLoader[T]) for_each(fn_each fn (batch &vtl.Tensor[T]) bool) int {
	mut count := 0
	for batch in dl.iter() {
		if !fn_each(batch) {
			break
		}
		count++
	}
	return count
}

// for_each_with_labels applies `fn` to each (features, labels) batch.
// Only works if the DataLoader was created with labels.
pub fn (dl &DataLoader[T]) for_each_with_labels(fn_each fn (features &vtl.Tensor[T], labels &vtl.Tensor[T]) bool) int {
	mut count := 0
	for i := 0; i < dl.len(); i++ {
		feat, lab := dl.batch_with_labels(i) or { break }
		if !fn_each(feat, lab) {
			break
		}
		count++
	}
	return count
}

// for_each_collated applies the custom collation callback to every batch and
// passes each result to `fn_each`. Callback errors stop iteration and return
// to the caller.
pub fn (dl &DataLoader[T]) for_each_collated(fn_each fn (batch CollatedBatch[T]) bool) !int {
	mut count := 0
	for i in 0 .. dl.len() {
		batch := dl.collated_batch(i)!
		if !fn_each(batch) {
			break
		}
		count++
	}
	return count
}

// split splits a DataLoader into train and validation DataLoaders.
// val_fraction must be finite and in the inclusive range [0, 1].
pub fn (dl &DataLoader[T]) split(val_fraction f64) !(&DataLoader[T], &DataLoader[T]) {
	if math.is_nan(val_fraction) || math.is_inf(val_fraction, 0) || val_fraction < 0.0
		|| val_fraction > 1.0 {
		return error('validation fraction must be finite and between 0 and 1')
	}
	n := dl.indices.len
	val_size := int(f64(n) * val_fraction)
	train_size := n - val_size

	mut train_indices := dl.indices[..train_size].clone()
	mut val_indices := dl.indices[train_size..].clone()

	train_dl := &DataLoader[T]{
		dataset:       dl.dataset
		labels:        dl.labels
		batch_size:    dl.batch_size
		shuffle:       false
		drop_last:     dl.drop_last
		seed:          dl.seed
		sampler:       &nn_data.SequentialSampler{}
		batch_sampler: &nn_data.FixedBatchSampler{}
		collate_fn:    dl.collate_fn
		indices:       train_indices
		epoch:         dl.epoch
	}

	val_dl := &DataLoader[T]{
		dataset:       dl.dataset
		labels:        dl.labels
		batch_size:    dl.batch_size
		shuffle:       false
		drop_last:     false
		seed:          dl.seed
		sampler:       &nn_data.SequentialSampler{}
		batch_sampler: &nn_data.FixedBatchSampler{}
		collate_fn:    dl.collate_fn
		indices:       val_indices
		epoch:         0
	}

	return train_dl, val_dl
}

// mini_batch_count returns the total number of batches in one epoch.
pub fn (dl &DataLoader[T]) mini_batch_count() int {
	return dl.len()
}

// LCG RNG — deterministic shuffle only, not cryptographically safe.
struct LcgRng {
mut:
	state u64
}

fn rng_from_seed(seed u64) LcgRng {
	return LcgRng{
		state: if seed == 0 { u64(1) } else { seed }
	}
}

fn (mut rng LcgRng) next() u64 {
	rng.state = rng.state * 6364136223846793005 + 1442695040888963407
	return rng.state
}
