module datasets

import encoding.csv
import vtl

pub const (
	mnist_test_url  = 'https://pjreddie.com/media/files/mnist_test.csv'
	mnist_train_url = 'https://pjreddie.com/media/files/mnist_train.csv'
)

[heap]
pub struct MnistDataset {
pub:
	train_features &vtl.Tensor
	train_labels   &vtl.Tensor
	test_features  &vtl.Tensor
	test_labels    &vtl.Tensor
}

pub fn load_mnist() ?&MnistDataset {
	train_features, train_labels := load_mnist_from_url(datasets.mnist_train_url) ?
	test_features, test_labels := load_mnist_from_url(datasets.mnist_test_url) ?

	return &MnistDataset{
		train_features: train_features
		train_labels: train_labels
		test_features: test_features
		test_labels: test_labels
	}
}

pub fn load_mnist_from_url(url string) ?(&vtl.Tensor, &vtl.Tensor) {
	mut labels := []int{}
	mut features := []f32{}

	content := load_dataset_from_url(url) ?

	mut parser := csv.new_reader(content)
	for {
		items := parser.read() or { break }
		labels << items[0].int()
		features << items[1..].map(it.f32())
	}

	mut lt := vtl.from_varray(labels, [labels.len])
	mut lft := vtl.zeros([lt.shape[0], 10])

	mut iter := lt.iterator()
	mut pos := iter.pos
	for _ in 0 .. lt.size() {
		if el := iter.next() {
			lft.set([pos, el as int], 1)
			pos = iter.pos
		}
	}

	return vtl.from_varray(features, [lt.shape[0], 10]), lft
}
