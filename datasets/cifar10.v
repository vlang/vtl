module datasets

import vtl
import os

pub const cifar10_base_url = 'https://www.cs.toronto.edu/~kriz/'
pub const cifar10_file = 'cifar-10-binary.tar.gz'

// Cifar10Dataset holds the CIFAR-10 dataset.
pub struct Cifar10Dataset {
pub:
	train_features &vtl.Tensor[f64] = unsafe { nil }
	train_labels   &vtl.Tensor[f64] = unsafe { nil }
	test_features  &vtl.Tensor[f64] = unsafe { nil }
	test_labels    &vtl.Tensor[f64] = unsafe { nil }
}

// Cifar10Config holds configuration for CIFAR-10 dataset loading.
pub struct Cifar10Config {
pub:
	channels    int = 3
	height      int = 32
	width       int = 32
	num_classes int = 10
}

// class_names returns the CIFAR-10 class names.
pub fn class_names() []string {
	return [
		'airplane',
		'automobile',
		'bird',
		'cat',
		'deer',
		'dog',
		'frog',
		'horse',
		'ship',
		'truck',
	]
}

fn download_cifar10(dataset string, baseurl string, file string, uncompressed_dir string) !string {
	return download_dataset(
		dataset:          dataset
		baseurl:          baseurl
		file:             file
		compressed:       true
		uncompressed_dir: uncompressed_dir
	)!
}

// load_cifar10_batch loads a single batch file from CIFAR-10 binary format.
fn load_cifar10_batch(path string, expected_count int) !([]f64, []int) {
	if !os.exists(path) {
		return error('CIFAR-10 batch not found: ${path}')
	}
	data := os.read_file(path)!
	mut labels := []int{len: expected_count}
	mut images := []f64{len: expected_count * 3072}
	for i := 0; i < expected_count; i++ {
		offset := i * 3073
		labels[i] = int(data[offset])
		for j := 0; j < 3072; j++ {
			images[i * 3072 + j] = f64(u8(data[offset + 1 + j])) / 255.0
		}
	}
	return images, labels
}

// load_cifar10_train_batches loads all 5 training batches.
fn load_cifar10_train_batches(data_dir string) !([]f64, []int) {
	mut all_images := []f64{len: 0}
	mut all_labels := []int{len: 0}
	for batch_num := 1; batch_num <= 5; batch_num++ {
		batch_path := os.join_path(data_dir, 'data_batch_${batch_num}.bin')
		images, labels := load_cifar10_batch(batch_path, 10000)!
		all_images << images
		all_labels << labels
	}
	return all_images, all_labels
}

// load_cifar10 loads the CIFAR-10 dataset.
// Returns train/test features (normalized [0,1]) and one-hot encoded labels.
pub fn load_cifar10() !Cifar10Dataset {
	cfg := Cifar10Config{}
	return load_cifar10_with_config(cfg)
}

// load_cifar10_with_config loads CIFAR-10 with custom configuration.
pub fn load_cifar10_with_config(cfg Cifar10Config) !Cifar10Dataset {
	dataset_path := download_cifar10('cifar10', cifar10_base_url, cifar10_file,
		'cifar-10-batches-bin')!

	// Load training data
	train_images, train_labels := load_cifar10_train_batches(dataset_path)!

	// Reshape training images to [N, C, H, W]
	train_features := vtl.from_1d(train_images)!.reshape([-1, cfg.channels, cfg.height, cfg.width])!

	// One-hot encode training labels
	mut train_labels_onehot := vtl.zeros[f64]([train_labels.len, cfg.num_classes])
	for i := 0; i < train_labels.len; i++ {
		class_idx := train_labels[i]
		if class_idx >= 0 && class_idx < cfg.num_classes {
			train_labels_onehot.set([i, class_idx], 1.0)
		}
	}
	train_labels_tensor :=
		vtl.from_1d(train_labels_onehot.to_array())!.reshape([-1, cfg.num_classes])!

	// Load test data
	test_path := os.join_path(dataset_path, 'test_batch.bin')
	test_images, test_labels := load_cifar10_batch(test_path, 10000)!

	// Reshape test images to [N, C, H, W]
	test_features := vtl.from_1d(test_images)!.reshape([-1, cfg.channels, cfg.height, cfg.width])!

	// One-hot encode test labels
	mut test_labels_onehot := vtl.zeros[f64]([test_labels.len, cfg.num_classes])
	for i := 0; i < test_labels.len; i++ {
		class_idx := test_labels[i]
		if class_idx >= 0 && class_idx < cfg.num_classes {
			test_labels_onehot.set([i, class_idx], 1.0)
		}
	}
	test_labels_tensor :=
		vtl.from_1d(test_labels_onehot.to_array())!.reshape([-1, cfg.num_classes])!

	return Cifar10Dataset{
		train_features: train_features
		train_labels:   train_labels_tensor
		test_features:  test_features
		test_labels:    test_labels_tensor
	}
}
