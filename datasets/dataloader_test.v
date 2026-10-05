module datasets

import vtl

fn test_contiguous_batches_are_zero_copy_views() ! {
	mut data := vtl.from_array([1.0, 2, 3, 4, 5, 6, 7, 8], [4, 2])!
	loader := new_data_loader[f64](data, DataLoaderConfig{
		batch_size: 2
		shuffle:    false
		drop_last:  false
	})
	mut batch := loader.batch(1) or { panic('expected second batch') }
	assert batch.shape == [2, 2]
	assert batch.to_array() == [5.0, 6, 7, 8]
	batch.set([0, 0], 50)
	assert data.get([2, 0]) == 50
}

fn test_non_contiguous_batches_copy_and_preserve_order() ! {
	mut data := vtl.from_array([1.0, 2, 3, 4, 5, 6, 7, 8], [4, 2])!
	mut loader := new_data_loader[f64](data, DataLoaderConfig{
		batch_size: 2
		shuffle:    false
		drop_last:  false
	})
	loader.indices = [2, 0, 3, 1]
	mut batch := loader.batch(0) or { panic('expected first batch') }
	assert batch.to_array() == [5.0, 6, 1, 2]
	batch.set([0, 0], 50)
	assert data.get([2, 0]) == 5
}

fn test_batch_with_labels_uses_matching_contiguous_views() ! {
	mut data := vtl.from_array([1.0, 2, 3, 4, 5, 6], [3, 2])!
	mut labels := vtl.from_array([10.0, 20, 30], [3])!
	loader := new_data_loader_with_labels[f64](data, labels, DataLoaderConfig{
		batch_size: 2
		shuffle:    false
		drop_last:  false
	})
	mut features_batch, mut labels_batch := loader.batch_with_labels(0) or {
		panic('expected a labeled batch')
	}
	assert features_batch.to_array() == [1.0, 2, 3, 4]
	assert labels_batch.to_array() == [10.0, 20]
	features_batch.set([0, 0], 100)
	labels_batch.set([0], 1000)
	assert data.get([0, 0]) == 100
	assert labels.get_nth(0) == 1000
}

fn test_batch_with_labels_gathers_shuffled_rows_as_copies() ! {
	mut data := vtl.from_array([1.0, 2, 3, 4, 5, 6], [3, 2])!
	mut labels := vtl.from_array([10.0, 20, 30], [3])!
	mut loader := new_data_loader_with_labels[f64](data, labels, DataLoaderConfig{
		batch_size: 2
		shuffle:    false
		drop_last:  false
	})
	loader.indices = [2, 0, 1]
	mut features_batch, mut labels_batch := loader.batch_with_labels(0) or {
		panic('expected a shuffled labeled batch')
	}
	assert features_batch.to_array() == [5.0, 6, 1, 2]
	assert labels_batch.to_array() == [30.0, 10]
	features_batch.set([0, 0], 100)
	labels_batch.set([0], 1000)
	assert data.get([2, 0]) == 5
	assert labels.get_nth(2) == 30
}
