module datasets

import rand
import vtl
import os

pub const (
	imdb_folder_name = 'aclImdb'
	imdb_file_name   = '${imdb_folder_name}_v1.tar.gz'
	imdb_base_url    = 'http://ai.stanford.edu/~amaas/data/sentiment/'
)

// ImdbDataset is a dataset for sentiment analysis.
pub struct ImdbDataset {
	paths []string
mut:
	batch_pos int
pub:
	@type      DatasetType
	batch_size int
}

// ImdbBatch is a batch of ImdbDataset.
pub struct ImdbBatch {
pub:
	features &vtl.Tensor<string>
	labels   &vtl.Tensor<int>
}

[params]
pub struct ImdbDatasetConfig {
	batch_size int = 32
}

// load_imdb loads the IMDB dataset iterator for a given split type.
pub fn load_imdb(set_type DatasetType, data ImdbDatasetConfig) ?&ImdbDataset {
	split := if set_type == .train { 'train' } else { 'test' }

	paths := download_dataset(
		dataset: 'imdb'
		baseurl: datasets.imdb_base_url
		extract: true
		tar: true
		urls_names: {
			datasets.imdb_file_name: datasets.imdb_folder_name
		}
	) ?

	mut split_paths := []string{}

	dataset_dir := paths[datasets.imdb_file_name]
	split_dir := os.join_path(dataset_dir, split)
	pos_dir := os.join_path(split_dir, 'pos')
	neg_dir := os.join_path(split_dir, 'neg')

	split_paths << os.walk_ext(pos_dir, '.txt')
	split_paths << os.walk_ext(neg_dir, '.txt')

	rand.shuffle(mut split_paths)

	return &ImdbDataset{
		@type: set_type
		batch_size: data.batch_size
		paths: split_paths
	}
}

// str is a string representation of the ImdbDataset.
pub fn (ds &ImdbDataset) str() string {
	mut res := []string{}
	res << 'vtl.datasets.ImdbDataset{'
	res << '    @type: ${ds.@type}'
	res << '    batch_size: $ds.batch_size'
	res << '}'
	return res.join('\n')
}

// next returns the next batch of the ImdbDataset.
pub fn (mut ds ImdbDataset) next() ?ImdbBatch {
	batch_size := ds.batch_size
	batch_pos := ds.batch_pos
	paths := ds.paths

	if batch_pos + batch_size > paths.len {
		return none
	}

	mut labels := []int{cap: batch_size}
	mut texts := []string{cap: batch_size}

	for path in paths[batch_pos..batch_pos + batch_size] {
		if !os.exists(path) {
			return none
		}

		content := os.read_file(path) ?
		file_name := os.file_name(path)
		label := file_name.split('_')[1]

		labels << label.int()
		texts << content
	}

	ds.batch_pos += batch_size

	mut lt := vtl.from_1d(labels)
	mut tt := vtl.from_1d(texts)

	return ImdbBatch{
		labels: lt
		features: tt
	}
}
