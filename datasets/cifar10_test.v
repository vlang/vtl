module datasets

import os
import math

fn write_cifar10_fixture_batch(path string, labels []u8, pixel u8) ! {
	mut bytes := []u8{len: labels.len * 3073}
	for i, label in labels {
		offset := i * 3073
		bytes[offset] = label
		for pixel_index in 0 .. 3072 {
			bytes[offset + 1 + pixel_index] = pixel
		}
	}
	os.write_file_array(path, bytes)!
}

fn test_cifar10_fixture_config_loads_shapes_labels_and_normalized_pixels() ! {
	dataset_dir := os.join_path(os.temp_dir(), 'vtl-cifar10-fixture-${os.getpid()}')
	os.mkdir_all(dataset_dir)!
	defer {
		os.rmdir_all(dataset_dir) or {}
	}
	write_cifar10_fixture_batch(os.join_path(dataset_dir, 'data_batch_1.bin'), [u8(2), 9], 127)!
	write_cifar10_fixture_batch(os.join_path(dataset_dir, 'test_batch.bin'), [u8(1), 8], 255)!

	dataset := load_cifar10_from_dir(dataset_dir, Cifar10Config{
		train_count: 2
		test_count:  1
	})!

	assert dataset.train_features.shape == [2, 3, 32, 32]
	assert dataset.train_labels.shape == [2, 10]
	assert dataset.test_features.shape == [1, 3, 32, 32]
	assert dataset.test_labels.shape == [1, 10]
	assert math.abs(dataset.train_features.get_nth(0) - 127.0 / 255.0) < 1e-12
	assert dataset.test_features.get_nth(3071) == 1.0
	assert dataset.train_labels.get([0, 2]) == 1.0
	assert dataset.train_labels.get([1, 9]) == 1.0
	assert dataset.test_labels.get([0, 1]) == 1.0
	assert dataset.train_labels.get([0, 0]) == 0.0
}

fn test_cifar10_default_config_and_class_names() {
	config := Cifar10Config{}
	assert config.channels == 3
	assert config.height == 32
	assert config.width == 32
	assert config.num_classes == 10
	assert config.train_count == 50000
	assert config.test_count == 10000
	assert class_names() == ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
		'ship', 'truck']
}

fn test_load_cifar10_batch_rejects_truncated_fixture() ! {
	path := os.join_path(os.temp_dir(), 'vtl-cifar10-truncated-${os.getpid()}.bin')
	os.write_file(path, 'short')!
	defer {
		os.rm(path) or {}
	}
	_, _ := load_cifar10_batch(path, 1) or {
		assert err.msg().contains('truncated')
		return
	}
	assert false, 'expected a truncated CIFAR-10 batch to fail'
}
