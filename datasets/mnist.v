module datasets

import vtl
import os

pub const (
	mnist_base_url          = 'http://yann.lecun.com/exdb/mnist/'
	mnist_train_images_file = 'train-images-idx3-ubyte.gz'
	mnist_train_labels_file = 'train-labels-idx1-ubyte.gz'
	mnist_test_images_file  = 't10k-images-idx3-ubyte.gz'
	mnist_test_labels_file  = 't10k-labels-idx1-ubyte.gz'
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
fn load_mnist_helper(filename string) !string {
	paths := download_dataset(
		dataset: 'mnist'
		baseurl: datasets.mnist_base_url
		extract: true
		urls_names: {
			filename: filename
		}
	)!

	path := paths[filename]
	uncompressed_path := path#[0..-3]
	return os.read_file(uncompressed_path)!
}

// load_mnist_features loads the MNIST features.
fn load_mnist_features(filename string) !&vtl.Tensor[u8] {
	content := load_mnist_helper(filename)!

	features := content[16..].bytes()

	return vtl.from_1d(features)!.reshape([-1, 28, 28])
}

// load_mnist_labels loads the MNIST labels.
fn load_mnist_labels(filename string) !&vtl.Tensor[u8] {
	content := load_mnist_helper(filename)!

	labels := content[8..].bytes()

	return vtl.from_1d(labels)!
}

// load_mnist loads the MNIST dataset.
pub fn load_mnist() !MnistDataset {
	train_features := load_mnist_features(datasets.mnist_train_images_file)!
	train_labels := load_mnist_labels(datasets.mnist_train_labels_file)!
	test_features := load_mnist_features(datasets.mnist_test_images_file)!
	test_labels := load_mnist_labels(datasets.mnist_test_labels_file)!

	return MnistDataset{
		train_features: train_features
		train_labels: train_labels
		test_features: test_features
		test_labels: test_labels
	}
}
