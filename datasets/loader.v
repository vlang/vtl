module datasets

import os
import crypto.sha1
import net.http
import vtl

pub enum DatasetType {
	train
	test
}

pub interface DatasetLoader {
	@type DatasetType
	batch_size int
	next() ?DatasetBatch
}

pub interface DatasetBatch {
	features &vtl.Tensor<f32>
	labels &vtl.Tensor<f32>
}

fn get_cache_dir(subdir string) string {
	cache_dir := os.cache_dir()
	return os.join_path(cache_dir, subdir)
}

fn load_dataset_from_url(url string) ?string {
	datasets_cache_dir := get_cache_dir('datasets')

	if !os.is_dir(datasets_cache_dir) {
		os.mkdir_all(datasets_cache_dir) ?
	}
	cache_file_name := sha1.hexhash(url)
	cache_file_path := os.join_path(datasets_cache_dir, cache_file_name)

	if os.is_file(cache_file_path) {
		return os.read_file(cache_file_path)
	}

	res := http.get(url) ?
	content := res.text

	os.write_file(cache_file_path, content) ?

	return content
}
