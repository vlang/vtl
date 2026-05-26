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
pub struct DataLoader[T] {
pub:
	dataset    &vtl.Tensor[T]
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

	// Collect indices for this batch
	batch_indices := dl.indices[start..end]

	// Fast path: single index → single view (no concatenation needed)
	if batch_indices.len == 1 {
		idx := batch_indices[0]
		return dl.dataset.slice([idx, idx + 1])!.view()
	}

	// Slow path: concatenate multiple rows into a contiguously allocated batch.
	// This is necessary because the dataset may be non-contiguous (e.g., a view
	// into a larger tensor). For CIFAR-10 with [N,C,H,W] tensors this is the
	// common case during training.
	mut result_shape := dl.dataset.shape.clone()
	result_shape[0] = batch_indices.len

	mut result := vtl.zeros[T](result_shape)
	row_size := dl.dataset.strides[0] // elements per row (C*H*W for CIFAR)
	for j, idx in batch_indices {
		// Copy row j from dataset at index `idx` into result[j, :]
		src_base := idx * row_size
		dst_base := j * row_size
		for k := 0; k < row_size; k++ {
			val := dl.dataset.get_nth(src_base + k)
			result.set_nth(dst_base + k, val)
		}
	}
	return result
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
	i := dl.indices.len - 1
	for i > 0 {
		j := int(rng.next() % u64(i + 1))
		dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		i--
	}
}

// new_data_loader creates a DataLoader from a dataset tensor.
// The dataset must have shape [N, ...] where N is the number of samples.
// Labels (if separate) should be passed as a second tensor and are indexed in lockstep.
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
// Reset the loader between epochs by calling `loader.reset()`.
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
// Returns the number of batches processed.
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

// split splits a DataLoader into train and validation DataLoaders.
// The validation fraction is taken from the end of the shuffled index array.
// Returns (train_loader, val_loader).
// Note: the returned loaders share the same dataset reference.
pub fn (dl &DataLoader[T]) split(val_fraction f64) (&DataLoader[T], &DataLoader[T]) {
	n := dl.indices.len
	val_size := int(f64(n) * val_fraction)
	train_size := n - val_size

	mut train_indices := dl.indices[..train_size].clone()
	mut val_indices := dl.indices[train_size..].clone()

	// Train loader: keep same shuffle/seed/epoch state if needed
	train_dl := &DataLoader[T]{
		dataset:    dl.dataset
		batch_size: dl.batch_size
		shuffle:    false // already shuffled at construction
		drop_last:  dl.drop_last
		seed:       dl.seed
		indices:    train_indices
		epoch:      dl.epoch
	}

	val_dl := &DataLoader[T]{
		dataset:    dl.dataset
		batch_size: dl.batch_size
		shuffle:    false
		drop_last:  false // always keep all validation samples
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

// simple RNG implementation using a LCG (Linear Congruential Generator).
// Not cryptographically safe — only for deterministic shuffling.
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
