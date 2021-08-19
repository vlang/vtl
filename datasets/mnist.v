module datasets

import encoding.csv
import vtl

pub const (
	mnist_test_url  = 'https://pjreddie.com/media/files/mnist_test.csv'
	mnist_train_url = 'https://pjreddie.com/media/files/mnist_train.csv'
)

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
	labels   &vtl.Tensor<f32>
}

pub fn load_mnist(set_type DatasetType, batch_size int) ?DatasetLoader {
	url := if set_type == .train { datasets.mnist_train_url } else { datasets.mnist_test_url }
	content := load_dataset_from_url(url) ?

	return DatasetLoader(&MnistDataset{
		@type: set_type
		batch_size: batch_size
		parser: csv.new_reader(content)
	})
}

pub fn (ds &MnistDataset) str() string {
	mut res := []string{}
	res << 'vtl.datasets.MnistDataset{'
	res << '    @type: ${ds.@type}'
	res << '    batch_size: $ds.batch_size'
	res << '}'
	return res.join('\n')
}

pub fn (mut ds MnistDataset) next() ?DatasetBatch {
	batch_size := ds.batch_size

	mut labels := []f32{cap: batch_size}
	mut features := []f32{cap: batch_size}

	for _ in 0 .. batch_size {
		items := ds.parser.read() or { break }
		labels << items[0].f32()
		features << items[1..].map(it.f32())
	}

	if labels.len == 0 {
		return none
	}

	mut lt := vtl.from_array(labels, [labels.len])
	mut lft := vtl.zeros<f32>([lt.shape[0], 10])

	mut iter := lt.iterator()
	mut pos := iter.pos
	for {
		el := iter.next() or { break }
		lft.set([pos, int(el)], 1)
		pos = iter.pos
	}

	ft := vtl.from_array(features, [features.len]).reshape([lt.shape[0], -1])

	return DatasetBatch(&MnistBatch{
		labels: lft
		features: ft
	})
}
