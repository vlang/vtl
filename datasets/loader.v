module datasets

import os
import crypto.sha1
import net.http
import szip
import vtl

pub enum DatasetType {
	train
	test
}

pub interface DatasetLoader {
	@type DatasetType
	batch_size int
mut:
	next() ?DatasetBatch
}

pub interface DatasetBatch {
	features &vtl.Tensor<f32>
	labels &vtl.Tensor<f32>
}

fn get_cache_dir(subdir ...string) string {
	mut cache_dir := os.cache_dir()
	$if datasets_dir ? {
		cache_dir = datasets_dir
	}
	return os.join_path(cache_dir, ...subdir)
}

struct RawDownload {
	url    string
	target string
}

fn load_from_url(data RawDownload) ?string {
	datasets_cache_dir := get_cache_dir('datasets')

	if !os.is_dir(datasets_cache_dir) {
		os.mkdir_all(datasets_cache_dir) ?
	}
	cache_file_name := sha1.hexhash(data.url)
	cache_file_path := if data.target == '' {
		os.join_path(datasets_cache_dir, cache_file_name)
	} else {
		data.target
	}

	if os.is_file(cache_file_path) {
		return os.read_file(cache_file_path)
	}

	res := http.get(data.url) ?
	content := res.text

	os.write_file(cache_file_path, content) ?

	return content
}

struct DatasetDownload {
	dataset    string
	baseurl    string
	extract    bool
	urls_names map[string]string
}

fn download_dataset(data DatasetDownload) ? {
	for path, filename in data.urls_names {
		dataset_dir := get_cache_dir('datasets', data.dataset)

		if !os.is_dir(dataset_dir) {
			os.mkdir_all(dataset_dir) ?
		}

		target := os.join_path(dataset_dir, filename)

		if os.exists(target) {
			$if debug ? {
				// we assume that the correct extraction process was done
				// before
				// @todo: check for extraction...
				println('$filename already exists')
			}
		} else {
			$if debug ? {
				println('Downloading $filename from $data.baseurl$path')
			}
			load_from_url(url: '$data.baseurl$path', target: target) ?
			if data.extract {
				$if debug ? {
					println('Extracting $target')
				}
				szip.extract_zip_to_dir(target, dataset_dir) ?
			}
		}
	}
}
