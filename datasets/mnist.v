module datasets

import encoding.csv
import vtl

pub const (
	mnist_test_url  = 'https://pjreddie.com/media/files/mnist_test.csv'
	mnist_train_url = 'https://pjreddie.com/media/files/mnist_train.csv'
)

pub enum DatasetType {
	train
	test
}

pub struct MnistDataset {
pub:
	@type      DatasetType
	batch_size int
mut:
	parser &csv.Reader
}

pub struct MnistBatch {
pub:
	features &vtl.Tensor<f32>
	labels   &vtl.Tensor<int>
}

pub fn load_mnist(set_type DatasetType, batch_size int) ?&MnistDataset {
	url := if set_type == .train { datasets.mnist_train_url } else { datasets.mnist_test_url }
	content := load_dataset_from_url(url) ?

	return &MnistDataset{
		@type: set_type
		batch_size: batch_size
		parser: csv.new_reader(content)
	}
}

pub fn (ds &MnistDataset) str() string {
	mut res := []string{}
	res << 'vtl.datasets.MnistDataset{'
	res << '    @type: ${ds.@type}'
	res << '    batch_size: $ds.batch_size'
	res << '}'
	return res.join('\n')
}

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

	mut lt := vtl.from_array(labels, [labels.len])
	mut lft := vtl.zeros<int>([lt.shape[0], 10])

	mut iter := lt.iterator()
	mut pos := iter.pos
	for {
		el := iter.next() or { break }
		lft.set([pos, el], 1)
		pos = iter.pos
	}

	return MnistBatch{
		labels: lft
		features: vtl.from_array(features, [lt.shape[0], 10])
	}
}
