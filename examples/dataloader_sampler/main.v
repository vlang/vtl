module main

import vtl
import vtl.datasets
import vtl.nn.data as nn_data

fn main() {
	features := vtl.from_2d([[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])!
	labels := vtl.from_1d([10.0, 11, 12, 13])!
	subset_sampler := nn_data.SubsetRandomSampler.new([1, 3])
	mut loader := datasets.new_data_loader_with_labels_and_sampler_checked[f64](features, labels,
		datasets.DataLoaderConfig{
			batch_size: 1
			shuffle:    false
			drop_last:  false
		}, subset_sampler)!
	for i in 0 .. loader.len() {
		batch, batch_labels := loader.batch_with_labels(i) or { continue }
		println('sample=${int(batch_labels.get_nth(0))}, features=${batch.to_array()}')
	}
	weighted_sampler := nn_data.WeightedRandomSampler.new([0.0, 1.0, 0.0, 3.0], 4, true)!
	weighted_loader := datasets.new_data_loader_with_labels_and_sampler_checked[f64](features,
		labels, datasets.DataLoaderConfig{
			batch_size: 4
			shuffle:    false
			drop_last:  false
		}, weighted_sampler)!
	_, weighted_labels := weighted_loader.batch_with_labels(0) or {
		panic('expected a weighted labeled batch')
	}
	println('weighted batch labels=${weighted_labels.to_array()}')
}
