module datasets

import vtl

// DatasetType is the type of a dataset.
pub enum DatasetType {
	train
	test
}

// DatasetLoader is an interface for loading datasets.
pub interface DatasetLoader {
	@type DatasetType
	batch_size int
mut:
	next() ?DatasetBatch
}

// @todo: Use generic type for DatasetBatch
pub interface DatasetBatch {
	labels &vtl.Tensor[int]
	features &vtl.Tensor[string]
}
