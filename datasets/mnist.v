module datasets

import encoding.csv
import os
import vtl

pub const (
	mnist_base_url   = 'https://pjreddie.com/media/files/'
	mnist_test_file  = 'mnist_test.csv'
	mnist_train_file = 'mnist_train.csv'
)

interface Reader {
mut:
	read() ?[]string
}

// MnistDataset is a dataset of MNIST handwritten digits.
pub struct MnistDataset {
pub:
	@type      DatasetType
	batch_size int
mut:
	parser Reader
}

// MnistBatch is a batch of MNIST handwritten digits.
pub struct MnistBatch {
pub:
	features &vtl.Tensor<u8>
	labels   &vtl.Tensor<u8>
}

[params]
pub struct MnistDatasetConfig {
	batch_size int = 32
}

// load_mnist returns a new MNIST iterator.
pub fn load_mnist(set_type DatasetType, data MnistDatasetConfig) !&MnistDataset {
	filename := if set_type == .train { datasets.mnist_train_file } else { datasets.mnist_test_file }

	paths := download_dataset(
		dataset: 'mnist'
		baseurl: datasets.mnist_base_url
		urls_names: {
			filename: filename
		}
	)!

	path := paths[filename]
	content := os.read_file(path)!

	return &MnistDataset{
		@type: set_type
		batch_size: data.batch_size
		parser: Reader(csv.new_reader(content))
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

	mut labels := []u8{cap: batch_size}
	mut features := []u8{cap: batch_size}

	for _ in 0 .. batch_size {
		items := ds.parser.read() or { break }
		labels << items[0].u8()
		features << items[1..].map(it.u8())
	}

	if labels.len == 0 {
		return none
	}

	mut lt := vtl.from_1d(labels)?
	mut lft := vtl.zeros<u8>([10])

	mut iter := lt.iterator()
	for {
		_, i := iter.next() or { break }
		lft.set(i, 1)
	}

	ft := vtl.from_array(features, [features.len])?.reshape([-1, 1, 32, 32])?

	return MnistBatch{
		labels: lft
		features: ft
	}
}
