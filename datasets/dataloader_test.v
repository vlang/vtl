module datasets

import vtl
import vtl.nn.data as nn_data
import math

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
	loader := new_data_loader_with_labels_checked[f64](data, labels, DataLoaderConfig{
		batch_size: 2
		shuffle:    false
		drop_last:  false
	})!
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
	mut loader := new_data_loader_with_labels_checked[f64](data, labels, DataLoaderConfig{
		batch_size: 2
		shuffle:    false
		drop_last:  false
	})!
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

fn test_checked_data_loader_rejects_mismatched_label_count() ! {
	dataset := vtl.from_2d([[1.0, 2], [3, 4], [5, 6]])!
	labels := vtl.from_1d([10.0, 20])!

	new_data_loader_with_labels_checked[f64](dataset, labels, DataLoaderConfig{
		batch_size: 2
		shuffle:    false
	}) or {
		assert err.msg().contains('same number of samples')
		return
	}
	assert false, 'mismatched sample counts must fail'
}

fn test_discontiguous_batches_preserve_column_major_logical_order() ! {
	dataset := vtl.from_array([1.0, 3, 5, 2, 4, 6], [3, 2], memory: .col_major)!
	mut loader := new_data_loader[f64](dataset, DataLoaderConfig{
		batch_size: 2
		shuffle:    false
	})
	loader.indices = [2, 0]
	batch := loader.batch(0) or { panic('expected gathered column-major batch') }
	assert batch.to_array() == [5.0, 6, 1, 2]
}

fn test_non_positive_batch_size_is_empty_and_safe() ! {
	dataset := vtl.from_2d([[1.0, 2], [3, 4]])!
	for batch_size in [0, -1] {
		loader := new_data_loader[f64](dataset, DataLoaderConfig{
			batch_size: batch_size
			shuffle:    false
		})
		assert loader.len() == 0
		assert loader.batch(0) == none
		assert loader.batch(-1) == none
	}
}

fn test_split_validates_validation_fraction() ! {
	dataset := vtl.from_1d([1.0, 2, 3, 4])!
	loader := new_data_loader[f64](dataset, DataLoaderConfig{
		batch_size: 2
		shuffle:    false
		drop_last:  false
	})

	train, validation := loader.split(0.25)!
	assert train.total_samples() == 3
	assert validation.total_samples() == 1
	all_training, empty_validation := loader.split(0.0)!
	assert all_training.total_samples() == 4
	assert empty_validation.total_samples() == 0
	empty_training, all_validation := loader.split(1.0)!
	assert empty_training.total_samples() == 0
	assert all_validation.total_samples() == 4

	for fraction in [-0.1, 1.1, math.nan(), math.inf(1)] {
		loader.split(fraction) or {
			assert err.msg().contains('validation fraction')
			continue
		}
		assert false, 'invalid validation fraction ${fraction} must fail'
	}
}

fn test_sampler_drives_tensor_batches_and_labels() ! {
	dataset := vtl.from_2d([[0.0], [1], [2], [3]])!
	labels := vtl.from_1d([10.0, 11, 12, 13])!
	sampler := nn_data.SubsetRandomSampler.new([1, 3])
	mut loader := new_data_loader_with_labels_and_sampler_checked[f64](dataset, labels,
		DataLoaderConfig{
			batch_size: 1
			shuffle:    false
			drop_last:  false
		}, sampler)!
	assert loader.total_samples() == 2
	mut selected_labels := []f64{}
	for i in 0 .. loader.len() {
		_, batch_labels := loader.batch_with_labels(i) or { panic('expected labeled batch') }
		selected_labels << batch_labels.get_nth(0)
	}
	selected_labels.sort()
	assert selected_labels == [11.0, 13.0]
	loader.reset_checked()!
	assert loader.total_samples() == 2
}

fn test_weighted_sampler_drives_tensor_batches() ! {
	dataset := vtl.from_2d([[0.0], [1], [2]])!
	labels := vtl.from_1d([10.0, 11, 12])!
	sampler := nn_data.WeightedRandomSampler.new([0.0, 1.0, 0.0], 5, true)!
	loader := new_data_loader_with_labels_and_sampler_checked[f64](dataset, labels,
		DataLoaderConfig{
			batch_size: 5
			shuffle:    false
			drop_last:  false
		}, sampler)!
	_, batch_labels := loader.batch_with_labels(0) or { panic('expected labeled batch') }
	assert batch_labels.to_array() == [11.0, 11, 11, 11, 11]
}

fn test_batch_sampler_controls_batch_boundaries() ! {
	dataset := vtl.from_2d([[0.0], [1], [2], [3]])!
	labels := vtl.from_1d([10.0, 11, 12, 13])!
	sequential_sampler := nn_data.Sampler(&nn_data.SequentialSampler{})
	sampler := nn_data.FixedBatchSampler.new(sequential_sampler, 3, false)!
	loader := new_data_loader_with_labels_and_batch_sampler_checked[f64](dataset, labels,
		DataLoaderConfig{
			batch_size: 99
			shuffle:    false
			drop_last:  false
		}, sampler)!
	assert loader.len() == 2
	first_batch := loader.batch(0) or { panic('expected first explicit batch') }
	last_batch := loader.batch(1) or { panic('expected final explicit batch') }
	assert first_batch.shape == [3, 1]
	assert last_batch.to_array() == [3.0]
	_, first_batch_labels := loader.batch_with_labels(0) or {
		panic('expected labeled explicit batch')
	}
	assert first_batch_labels.to_array() == [10.0, 11, 12]
}

fn test_batch_sampler_drop_last_and_validation() ! {
	dataset := vtl.from_1d([0.0, 1, 2, 3, 4])!
	sequential_sampler := nn_data.Sampler(&nn_data.SequentialSampler{})
	sampler := nn_data.FixedBatchSampler.new(sequential_sampler, 2, true)!
	loader := new_data_loader_with_batch_sampler_checked[f64](dataset, DataLoaderConfig{
		batch_size: 1
		shuffle:    false
		drop_last:  false
	}, sampler)!
	assert loader.len() == 2
	assert loader.total_samples() == 4
	nn_data.FixedBatchSampler.new(sequential_sampler, 0, false) or {
		assert err.msg().contains('batch_size')
		return
	}
	assert false, 'non-positive batch size must fail'
}
