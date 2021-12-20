module datasets

import encoding.csv
import os
import vtl

pub const (
	mnist_base_url   = 'https://pjreddie.com/media/files/'
	mnist_test_file  = 'mnist_test.csv'
	mnist_train_file = 'mnist_train.csv'
)

// MnistDataset is a dataset of MNIST handwritten digits.
pub struct MnistDataset {
pub:
	@type      DatasetType
	batch_size int
mut:
	parser &csv.Reader
}

// MnistBatch is a batch of MNIST handwritten digits.
pub struct MnistBatch {
pub:
	features &vtl.Tensor<f32>
	labels   &vtl.Tensor<int>
}

[params]
pub struct MnistDatasetConfig {
	batch_size int = 32
}

// load_mnist returns a new MNIST iterator.
pub fn load_mnist(set_type DatasetType, data MnistDatasetConfig) ?&MnistDataset {
	filename := if set_type == .train { datasets.mnist_train_file } else { datasets.mnist_test_file }

	paths := download_dataset(
		dataset: 'mnist'
		baseurl: datasets.mnist_base_url
		urls_names: {
			filename: filename
		}
	) ?

	path := paths[filename]
	content := os.read_file(path) ?

	return &MnistDataset{
		@type: set_type
		batch_size: data.batch_size
		parser: csv.new_reader(content)
	}
}

// str is a string representation of the MnistDataset.
pub fn (ds &MnistDataset) str() string {
	mut res := []string{}
	res << 'vtl.datasets.MnistDataset{'
	res << '    @type: ${ds.@type}'
	res << '    batch_size: $ds.batch_size'
	res << '}'
	return res.join('\n')
}

// next returns the next batch of MNIST handwritten digits.
pub fn (mut ds MnistDataset) next() ?MnistBatch {
	batch_size := ds.batch_size

	mut labels := []int{cap: batch_size}
	mut features := []f32{cap: batch_size}

	for _ in 0 .. batch_size {
		items := ds.parser.read() or { break }
		labels << items[0].int()
		features << items[1..].map(it.f32())
	}

	if labels.len == 0 {
		return none
	}

	mut lt := vtl.from_1d(labels)
	mut lft := vtl.zeros<int>([lt.shape[0], 10])

	mut iter := lt.iterator()
	for {
		elem, pos := iter.next() or { break }
		lft.set([pos, int(elem)], 1)
	}

	ft := vtl.from_array(features, [features.len]).reshape([-1, 1, 32, 32])

	return MnistBatch{
		labels: lft
		features: ft
	}
}
