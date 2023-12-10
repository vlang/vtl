module datasets

import vtl
import os

pub const imdb_folder_name = 'aclImdb'
pub const imdb_file_name = '${imdb_folder_name}_v1.tar.gz'
pub const imdb_base_url = 'http://ai.stanford.edu/~amaas/data/sentiment/'

// ImdbDataset is a dataset for sentiment analysis.
pub struct ImdbDataset {
pub:
	train_features &vtl.Tensor[string] = unsafe { nil }
	train_labels   &vtl.Tensor[int]    = unsafe { nil }
	test_features  &vtl.Tensor[string] = unsafe { nil }
	test_labels    &vtl.Tensor[int]    = unsafe { nil }
}

// load_imdb_helper loads the IMDB dataset for a given split.
fn load_imdb_helper(split string) !(&vtl.Tensor[string], &vtl.Tensor[int]) {
	paths := download_dataset(
		dataset: 'imdb'
		baseurl: datasets.imdb_base_url
		extract: true
		tar: true
		urls_names: {
			datasets.imdb_file_name: datasets.imdb_folder_name
		}
	)!

	mut split_paths := []string{}

	dataset_dir := paths[datasets.imdb_file_name]
	split_dir := os.join_path(dataset_dir, split)
	pos_dir := os.join_path(split_dir, 'pos')
	neg_dir := os.join_path(split_dir, 'neg')

	split_paths << os.walk_ext(pos_dir, '.txt')
	split_paths << os.walk_ext(neg_dir, '.txt')

	mut labels := []int{cap: split_paths.len}
	mut texts := []string{cap: split_paths.len}

	for path in split_paths {
		if !os.exists(path) {
			return error('file does not exist')
		}

		content := os.read_file(path)!
		file_name := os.file_name(path)
		label := file_name.split('_')[1]

		labels << label.int()
		texts << content
	}

	mut lt := vtl.from_1d(labels)!
	mut tt := vtl.from_1d(texts)!

	return tt, lt
}

// load_imdb loads the IMDB dataset.
pub fn load_imdb() !ImdbDataset {
	train_features, train_labels := load_imdb_helper('train')!
	test_features, test_labels := load_imdb_helper('test')!

	return ImdbDataset{
		train_features: train_features
		train_labels: train_labels
		test_features: test_features
		test_labels: test_labels
	}
}
