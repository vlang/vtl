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
	train_features &vtl.Tensor[u8]
	train_labels   &vtl.Tensor[u8]
	test_features  &vtl.Tensor[u8]
	test_labels    &vtl.Tensor[u8]
}

// load_mnist_helper loads the MNIST dataset from the given filename.
pub fn load_mnist_helper(filename string) !(&vtl.Tensor[u8], &vtl.Tensor[u8]) {
	paths := download_dataset(
		dataset: 'mnist'
		baseurl: datasets.mnist_base_url
		urls_names: {
			filename: filename
		}
	)!

	path := paths[filename]
	content := os.read_file(path)!
	mut parser := csv.new_reader(content)

	mut labels := []int{}
	mut features := []u8{}

	for {
		items := parser.read() or { break }
		labels << items[0].int()
		features << items[1..].map(it.u8())
	}

	mut lt := vtl.from_1d(labels)!
	mut lft := vtl.zeros[u8]([lt.shape[0], 10])

	mut iter := lt.iterator()
	for {
		val, i := iter.next() or { break }
		mut next_index := i.clone()
		next_index << val
		lft.set(next_index, 1)
	}

	ft := vtl.from_1d(features)!.reshape([-1, 28, 28])!

	return ft, lft
}

// load_mnist loads the MNIST dataset.
pub fn load_mnist() !MnistDataset {
	train_features, train_labels := load_mnist_helper(datasets.mnist_train_file)!
	test_features, test_labels := load_mnist_helper(datasets.mnist_test_file)!

	return MnistDataset{
		train_features: train_features
		train_labels: train_labels
		test_features: test_features
		test_labels: test_labels
	}
}
