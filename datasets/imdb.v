module datasets

import vtl
import os

// imdb_file_name is a public constant used by this module.
pub const imdb_file_name = 'aclImdb_v1.tar.gz'
// imdb_base_url is a public constant used by this module.
pub const imdb_base_url = 'http://ai.stanford.edu/~amaas/data/sentiment/'

const imdb_label_files_count = 12500

// ImdbConfig configures the number of balanced examples returned per split.
@[params]
pub struct ImdbConfig {
pub:
	train_count int = 25000
	test_count  int = 25000
}

// ImdbDataset is a dataset for sentiment analysis.
pub struct ImdbDataset {
pub:
	train_features &vtl.Tensor[string] = unsafe { nil }
	train_labels   &vtl.Tensor[int]    = unsafe { nil }
	test_features  &vtl.Tensor[string] = unsafe { nil }
	test_labels    &vtl.Tensor[int]    = unsafe { nil }
}

fn imdb_split_paths(dataset_path string, split string, count int) ![]string {
	if count < 1 || count > imdb_label_files_count * 2 {
		return error('IMDB ${split} count must be between 1 and ${imdb_label_files_count * 2}')
	}
	split_dir := os.join_path(dataset_path, split)
	pos_dir := os.join_path(split_dir, 'pos')
	neg_dir := os.join_path(split_dir, 'neg')

	mut pos_paths := os.walk_ext(pos_dir, '.txt')
	mut neg_paths := os.walk_ext(neg_dir, '.txt')
	pos_count := (count + 1) / 2
	neg_count := count / 2
	if pos_paths.len < pos_count || neg_paths.len < neg_count {
		return error('invalid cached IMDB ${split} split: got ${pos_paths.len} positive and ${neg_paths.len} negative files, need ${pos_count} positive and ${neg_count} negative files')
	}
	pos_paths.sort()
	neg_paths.sort()

	mut split_paths := []string{cap: count}
	split_paths << pos_paths[..pos_count]
	split_paths << neg_paths[..neg_count]
	return split_paths
}

fn imdb_review_label(path string) !int {
	match os.base(os.dir(path)) {
		'pos' { return 1 }
		'neg' { return 0 }
		else { return error('invalid IMDB review label directory: ${path}') }
	}
}

fn load_imdb_split(dataset_path string, split string, count int) !(&vtl.Tensor[string], &vtl.Tensor[int]) {
	split_paths := imdb_split_paths(dataset_path, split, count)!

	mut labels := []int{cap: split_paths.len}
	mut texts := []string{cap: split_paths.len}

	for path in split_paths {
		if !os.exists(path) {
			return error('file does not exist')
		}

		content := os.read_file(path)!
		labels << imdb_review_label(path)!
		texts << content
	}

	mut lt := vtl.from_1d(labels)!
	mut tt := vtl.from_1d(texts)!

	return tt, lt
}

// load_imdb loads the IMDB dataset.
pub fn load_imdb() !ImdbDataset {
	return load_imdb_with_config(ImdbConfig{})
}

// load_imdb_with_config loads a balanced subset of the IMDB dataset.
pub fn load_imdb_with_config(cfg ImdbConfig) !ImdbDataset {
	if cfg.train_count < 1 || cfg.train_count > imdb_label_files_count * 2 {
		return error('IMDB train_count must be between 1 and ${imdb_label_files_count * 2}')
	}
	if cfg.test_count < 1 || cfg.test_count > imdb_label_files_count * 2 {
		return error('IMDB test_count must be between 1 and ${imdb_label_files_count * 2}')
	}
	mut dataset_path := download_dataset(
		dataset:          'imdb'
		baseurl:          imdb_base_url
		compressed:       true
		uncompressed_dir: 'aclImdb'
		file:             imdb_file_name
	)!
	return load_imdb_from_dir(dataset_path, cfg) or {
		if !err.msg().contains('invalid cached IMDB') {
			return err
		}
		os.rmdir_all(dataset_path)!
		dataset_path = download_dataset(
			dataset:          'imdb'
			baseurl:          imdb_base_url
			compressed:       true
			uncompressed_dir: 'aclImdb'
			file:             imdb_file_name
		)!
		load_imdb_from_dir(dataset_path, cfg)!
	}
}

fn load_imdb_from_dir(dataset_path string, cfg ImdbConfig) !ImdbDataset {
	train_features, train_labels := load_imdb_split(dataset_path, 'train', cfg.train_count)!
	test_features, test_labels := load_imdb_split(dataset_path, 'test', cfg.test_count)!

	return ImdbDataset{
		train_features: train_features
		train_labels:   train_labels
		test_features:  test_features
		test_labels:    test_labels
	}
}
