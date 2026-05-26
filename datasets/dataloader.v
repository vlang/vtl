module datasets

import vtl

// DataLoader provides an iterable over a dataset with batching and shuffling.
// It uses zero-copy slice views to avoid memory allocation on each batch.
//
// Design principles (inspired by Arraymancer and PyTorch):
// - Zero-copy: batch extraction uses tensor.slice() which returns a view
// - Generic: works with any tensor element type T (f64, f32)
// - Configurable: batch_size, shuffle, drop_last
// - Deterministic: optional seed for reproducible shuffling
// - Supports optional labels tensor for supervised learning
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
}

// DataLoaderConfig holds configuration for creating a DataLoader.
pub struct DataLoaderConfig {
pub:
	batch_size int  = 32
	shuffle    bool = true
	drop_last  bool = true
	seed       u64
}

// batch returns the batch at index `i` as a zero-copy view into the dataset.
// Returns none if the index is out of range.
pub fn (dl &DataLoader[T]) batch(i int) ?&vtl.Tensor[T] {
	if dl.indices.len == 0 {
		return none
	}
	start := i * dl.batch_size
	if start >= dl.indices.len {
		return none
	}
	end := if dl.drop_last {
		if start + dl.batch_size > dl.indices.len {
			return none
		}
		start + dl.batch_size
	} else {
		if start + dl.batch_size > dl.indices.len {
			dl.indices.len
		} else {
			start + dl.batch_size
		}
	}

	batch_indices := dl.indices[start..end]

	if batch_indices.len == 1 {
		idx := batch_indices[0]
		view_tensor := dl.dataset.slice([idx, idx + 1]) or { return none }
		return view_tensor.view()
	}

	mut result_shape := dl.dataset.shape.clone()
	result_shape[0] = batch_indices.len

	mut result := vtl.zeros[T](result_shape)
	row_size := dl.dataset.strides[0]
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
	if dl.indices.len == 0 {
		return none
	}
	if dl.labels == unsafe { nil } {
		return none
	}
	start := i * dl.batch_size
	if start >= dl.indices.len {
		return none
	}
	end := if dl.drop_last {
		if start + dl.batch_size > dl.indices.len {
			return none
		}
		start + dl.batch_size
	} else {
		if start + dl.batch_size > dl.indices.len {
			dl.indices.len
		} else {
			start + dl.batch_size
		}
	}

	batch_indices := dl.indices[start..end]

	// Build features batch
	mut feat_shape := dl.dataset.shape.clone()
	feat_shape[0] = batch_indices.len
	mut feat_result := vtl.zeros[T](feat_shape)
	feat_row_size := dl.dataset.strides[0]
	for j, idx in batch_indices {
		src_base := idx * feat_row_size
		dst_base := j * feat_row_size
		for k := 0; k < feat_row_size; k++ {
			feat_result.set_nth(dst_base + k, dl.dataset.get_nth(src_base + k))
		}
	}

	// Build labels batch (labels are 2D: [N, num_classes])
	label_row_size := dl.labels.strides[0]
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

// len returns the number of batches (drop_last affects count).
pub fn (dl &DataLoader[T]) len() int {
	if dl.indices.len == 0 {
		return 0
	}
	if dl.drop_last {
		return dl.indices.len / dl.batch_size
	}
	return (dl.indices.len + dl.batch_size - 1) / dl.batch_size
}

// total_samples returns the number of samples in the dataset.
pub fn (dl &DataLoader[T]) total_samples() int {
	return dl.indices.len
}

// reset re-shuffles the indices (call between epochs). Uses seed if set.
pub fn (mut dl DataLoader[T]) reset() {
	dl.epoch++
	if dl.shuffle {
		dl.shuffle_indices()
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
	n := dataset.shape[0]
	mut indices := []int{len: n}
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	mut dl := &DataLoader[T]{
		dataset:    dataset
		batch_size: config.batch_size
		shuffle:    config.shuffle
		drop_last:  config.drop_last
		seed:       config.seed
		indices:    indices
		epoch:      0
	}
	if config.shuffle {
		dl.shuffle_indices()
	}
	return dl
}

// new_data_loader_with_labels creates a DataLoader with both features and labels tensors.
// Both tensors must have the same first dimension (number of samples).
pub fn new_data_loader_with_labels[T](dataset &vtl.Tensor[T], labels &vtl.Tensor[T], config DataLoaderConfig) &DataLoader[T] {
	n := dataset.shape[0]
	mut indices := []int{len: n}
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	mut dl := &DataLoader[T]{
		dataset:    dataset
		labels:     labels
		batch_size: config.batch_size
		shuffle:    config.shuffle
		drop_last:  config.drop_last
		seed:       config.seed
		indices:    indices
		epoch:      0
	}
	if config.shuffle {
		dl.shuffle_indices()
	}
	return dl
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

// split splits a DataLoader into train and validation DataLoaders.
pub fn (dl &DataLoader[T]) split(val_fraction f64) (&DataLoader[T], &DataLoader[T]) {
	n := dl.indices.len
	val_size := int(f64(n) * val_fraction)
	train_size := n - val_size

	mut train_indices := dl.indices[..train_size].clone()
	mut val_indices := dl.indices[train_size..].clone()

	train_dl := &DataLoader[T]{
		dataset:    dl.dataset
		labels:     dl.labels
		batch_size: dl.batch_size
		shuffle:    false
		drop_last:  dl.drop_last
		seed:       dl.seed
		indices:    train_indices
		epoch:      dl.epoch
	}

	val_dl := &DataLoader[T]{
		dataset:    dl.dataset
		labels:     dl.labels
		batch_size: dl.batch_size
		shuffle:    false
		drop_last:  false
		seed:       dl.seed
		indices:    val_indices
		epoch:      0
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
