module datasets

import vtl
import os

pub const imdb_file_name = 'aclImdb_v1.tar.gz'
pub const imdb_base_url = 'http://ai.stanford.edu/~amaas/data/sentiment/'

const imdb_label_files_count = 12500

// ImdbDataset is a dataset for sentiment analysis.
pub struct ImdbDataset {
pub:
	train_features &vtl.Tensor[string] = unsafe { nil }
	train_labels   &vtl.Tensor[int]    = unsafe { nil }
	test_features  &vtl.Tensor[string] = unsafe { nil }
	test_labels    &vtl.Tensor[int]    = unsafe { nil }
}

fn imdb_split_paths(dataset_path string, split string) ![]string {
	split_dir := os.join_path(dataset_path, split)
	pos_dir := os.join_path(split_dir, 'pos')
	neg_dir := os.join_path(split_dir, 'neg')

	pos_paths := os.walk_ext(pos_dir, '.txt')
	neg_paths := os.walk_ext(neg_dir, '.txt')
	if pos_paths.len != imdb_label_files_count || neg_paths.len != imdb_label_files_count {
		return error('invalid cached IMDB ${split} split: got ${pos_paths.len} positive and ${neg_paths.len} negative files, expected ${imdb_label_files_count} each')
	}

	mut split_paths := []string{cap: pos_paths.len + neg_paths.len}
	split_paths << pos_paths
	split_paths << neg_paths
	return split_paths
}

// load_imdb_helper loads the IMDB dataset for a given split.
fn load_imdb_helper(split string) !(&vtl.Tensor[string], &vtl.Tensor[int]) {
	mut dataset_path := download_dataset(
		dataset:          'imdb'
		baseurl:          imdb_base_url
		compressed:       true
		uncompressed_dir: 'aclImdb'
		file:             imdb_file_name
	)!

	split_paths := imdb_split_paths(dataset_path, split) or {
		os.rmdir_all(dataset_path)!
		dataset_path = download_dataset(
			dataset:          'imdb'
			baseurl:          imdb_base_url
			compressed:       true
			uncompressed_dir: 'aclImdb'
			file:             imdb_file_name
		)!
		imdb_split_paths(dataset_path, split)!
	}

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
		train_labels:   train_labels
		test_features:  test_features
		test_labels:    test_labels
	}
}
