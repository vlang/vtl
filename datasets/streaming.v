module datasets

import vtl

// StreamSample contains one feature sample and optional labels. Set finished
// to true to signal exhaustion; tensor fields are ignored in that result.
pub struct StreamSample[T] {
pub:
	features &vtl.Tensor[T] = unsafe { nil }
	labels   &vtl.Tensor[T] = unsafe { nil }
	finished bool
}

// IterableDataset produces samples on demand and owns its source state.
// Implement reset when the source can be replayed; otherwise return an error.
pub interface IterableDataset[T] {
	// next_sample returns one sample, a finished marker, or a source error.
mut:
	next_sample() !StreamSample[T]
	// reset rewinds the source or reports that replay is unsupported.
	reset() !
}

// StreamBatch contains one dense batch produced from an iterable source.
pub struct StreamBatch[T] {
pub:
	available bool
	features  &vtl.Tensor[T] = unsafe { nil }
	labels    &vtl.Tensor[T] = unsafe { nil }
}

// IterableDataLoader batches an IterableDataset without materializing it.
pub struct IterableDataLoader[T] {
pub:
	batch_size int
	drop_last  bool
mut:
	dataset   IterableDataset[T]
	exhausted bool
}

// new_iterable_data_loader creates a loader over a source that yields equally
// shaped feature samples and, optionally, equally shaped labels.
pub fn new_iterable_data_loader[T](dataset IterableDataset[T], batch_size int, drop_last bool) !&IterableDataLoader[T] {
	if batch_size <= 0 {
		return error('batch_size must be positive')
	}
	return &IterableDataLoader[T]{
		dataset:    dataset
		batch_size: batch_size
		drop_last:  drop_last
	}
}

// next_batch reads at most batch_size samples and stacks them into dense
// tensors. It returns available=false after exhaustion; source errors and
// inconsistent sample shapes are returned to the caller.
pub fn (mut loader IterableDataLoader[T]) next_batch() !StreamBatch[T] {
	if loader.exhausted {
		return StreamBatch[T]{}
	}
	mut feature_samples := [][]T{cap: loader.batch_size}
	mut label_samples := [][]T{cap: loader.batch_size}
	mut feature_shape := []int{}
	mut label_shape := []int{}
	mut has_labels := false

	for _ in 0 .. loader.batch_size {
		sample := loader.dataset.next_sample()!
		if sample.finished {
			loader.exhausted = true
			if feature_samples.len == 0 || loader.drop_last {
				return StreamBatch[T]{}
			}
			break
		}
		if sample.features == unsafe { nil } {
			return error('iterable dataset returned a sample without features')
		}
		sample_has_labels := sample.labels != unsafe { nil }
		if feature_samples.len == 0 {
			has_labels = sample_has_labels
			feature_shape = sample.features.shape.clone()
			if has_labels {
				label_shape = sample.labels.shape.clone()
			}
		} else if sample_has_labels != has_labels {
			return error('iterable dataset must provide labels consistently')
		} else if sample.features.shape != feature_shape {
			return error('iterable dataset feature sample has an inconsistent shape')
		} else if has_labels && sample.labels.shape != label_shape {
			return error('iterable dataset label sample has an inconsistent shape')
		}
		// Copy immediately so datasets can safely reuse a mutable sample buffer.
		feature_samples << sample.features.to_array()
		if sample_has_labels {
			label_samples << sample.labels.to_array()
		}
	}

	features := stack_stream_values(feature_samples, feature_shape, 'feature')!
	if has_labels {
		labels := stack_stream_values(label_samples, label_shape, 'label')!
		return StreamBatch[T]{
			available: true
			features:  features
			labels:    labels
		}
	}
	return StreamBatch[T]{
		available: true
		features:  features
	}
}

// reset rewinds the underlying source. Sources that cannot replay should
// return an error from their reset implementation.
pub fn (mut loader IterableDataLoader[T]) reset() ! {
	loader.dataset.reset()!
	loader.exhausted = false
}

fn stack_stream_values[T](samples [][]T, sample_shape []int, kind string) !&vtl.Tensor[T] {
	if samples.len == 0 {
		return error('cannot stack an empty stream ${kind} batch')
	}
	row_size := samples[0].len
	for i, sample in samples {
		if sample.len != row_size {
			return error('iterable dataset ${kind} sample ${i} has an inconsistent shape')
		}
	}
	mut output_shape := []int{len: sample_shape.len + 1}
	output_shape[0] = samples.len
	for i, dimension in sample_shape {
		output_shape[i + 1] = dimension
	}
	mut output := vtl.zeros[T](output_shape)
	for sample_index, sample in samples {
		for i in 0 .. row_size {
			output.set_nth(sample_index * row_size + i, sample[i])
		}
	}
	return output
}
