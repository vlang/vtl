module main

import vtl
import vtl.datasets

fn main() {
	features := vtl.from_1d([1.0, 2, 3]) or { panic(err) }
	labels := vtl.from_1d([0.0, 1, 1]) or { panic(err) }
	mut loader := datasets.new_data_loader_with_labels_checked[f64](features, labels,
		datasets.DataLoaderConfig{
			batch_size: 2
			shuffle:    false
			drop_last:  false
		}) or {
		panic(err)
	}
	loader.indices = [2, 0, 1]
	factor := 10.0
	loader.set_collate_fn(fn [factor] (dataset &vtl.Tensor[f64], labels &vtl.Tensor[f64], indices []int) !datasets.CollatedBatch[f64] {
		mut collated_features := vtl.zeros[f64]([indices.len])
		mut collated_labels := vtl.zeros[f64]([indices.len])
		for i, sample_index in indices {
			collated_features.set_nth(i, dataset.get_nth(sample_index) * factor)
			collated_labels.set_nth(i, labels.get_nth(sample_index))
		}
		return datasets.CollatedBatch[f64]{
			features: collated_features
			labels:   collated_labels
		}
	})
	batch := loader.collated_batch(0) or { panic(err) }
	println('scaled features: ${batch.features.to_array()}')
	println('labels: ${batch.labels.to_array()}')
}
