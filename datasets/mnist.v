module datasets

import vtl
import os

pub const mnist_base_url = 'https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/'
pub const mnist_train_images_file = 'train-images-idx3-ubyte.gz'
pub const mnist_train_labels_file = 'train-labels-idx1-ubyte.gz'
pub const mnist_test_images_file = 't10k-images-idx3-ubyte.gz'
pub const mnist_test_labels_file = 't10k-labels-idx1-ubyte.gz'

// MnistDataset is a dataset of MNIST handwritten digits.
pub struct MnistDataset {
pub:
	train_features &vtl.Tensor[u8] = unsafe { nil }
	train_labels   &vtl.Tensor[u8] = unsafe { nil }
	test_features  &vtl.Tensor[u8] = unsafe { nil }
	test_labels    &vtl.Tensor[u8] = unsafe { nil }
}

// load_mnist_helper loads the MNIST dataset from the given filename.
fn load_mnist_helper(file string) !string {
	dataset_path := download_dataset(
		dataset: 'mnist'
		baseurl: datasets.mnist_base_url
		compressed: true
		uncompressed_dir: file.all_before_last('.')
		file: file
	)!

	return os.read_file(dataset_path)!
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
