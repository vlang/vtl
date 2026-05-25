module datasets

import os
import vtl

pub const iris_file_name = 'iris.data'
pub const iris_base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/'

pub const iris_features_count = 4

// IrisDataset is the classic Iris classification dataset.
//
// Features are shape [150, 4] with columns:
// - sepal length
// - sepal width
// - petal length
// - petal width
//
// Labels are encoded as:
// - 0 => Iris-setosa
// - 1 => Iris-versicolor
// - 2 => Iris-virginica
pub struct IrisDataset {
pub:
	features &vtl.Tensor[f64] = unsafe { nil }
	labels   &vtl.Tensor[int] = unsafe { nil }
}

fn iris_label_to_int(label string) !int {
	return match label {
		'Iris-setosa' { 0 }
		'Iris-versicolor' { 1 }
		'Iris-virginica' { 2 }
		else { return error('unknown iris label: ${label}') }
	}
}

// load_iris downloads (if needed) and parses the UCI Iris dataset.
pub fn load_iris() !IrisDataset {
	dataset_path := download_dataset(
		dataset:          'iris'
		baseurl:          iris_base_url
		compressed:       false
		uncompressed_dir: iris_file_name
		file:             iris_file_name
	)!

	content := os.read_file(dataset_path)!
	lines := content.split_into_lines()

	mut features := []f64{}
	mut labels := []int{}

	for line in lines {
		trimmed := line.trim_space()
		if trimmed == '' {
			continue
		}

		parts := trimmed.split(',')
		if parts.len != 5 {
			return error('invalid iris row: expected 5 columns, got ${parts.len}')
		}

		for i in 0 .. iris_features_count {
			features << parts[i].f64()
		}

		labels << iris_label_to_int(parts[4])!
	}

	if labels.len == 0 {
		return error('iris dataset is empty')
	}

	if features.len != labels.len * iris_features_count {
		return error('invalid iris tensor shape: features=${features.len}, labels=${labels.len}')
	}

	features_tensor := vtl.from_1d(features)!.reshape([-1, iris_features_count])!
	labels_tensor := vtl.from_1d(labels)!

	return IrisDataset{
		features: features_tensor
		labels:   labels_tensor
	}
}
