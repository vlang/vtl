module datasets

import os
import compress.gzip
import crypto.sha1
import net.http

fn get_cache_dir(subdir ...string) string {
	mut cache_dir := os.cache_dir()
	$if datasets_dir ? {
		cache_dir = datasets_dir
	}
	return os.join_path(cache_dir, ...subdir)
}

@[params]
struct RawDownload {
	url    string
	target string
}

fn load_from_url(data RawDownload) ! {
	datasets_cache_dir := get_cache_dir('datasets')
	if !os.is_dir(datasets_cache_dir) {
		os.mkdir_all(datasets_cache_dir)!
	}
	cache_file_name := sha1.hexhash(data.url)
	cache_file_path := if data.target == '' {
		os.join_path(datasets_cache_dir, cache_file_name)
	} else {
		data.target
	}
	if os.is_file(cache_file_path) {
		return
	}
	http.download_file(data.url, cache_file_path)!
}

@[params]
struct DatasetDownload {
	dataset          string
	baseurl          string
	file             string
	compressed       bool
	uncompressed_dir string @[required]
}

fn download_dataset(data DatasetDownload) !string {
	dataset_dir := os.real_path(get_cache_dir('datasets', data.dataset))

	// Handle extensions like `*.tar.gz`.
	exts := os.file_name(data.file).rsplit_nth('.', 3)
	is_tar := exts[0] == 'tar' || (exts.len > 1 && exts[1] == 'tar')

	target := os.join_path(dataset_dir, data.file)
	if os.exists(target) {
		$if debug ? {
			println('${data.file} already exists')
		}
	} else {
		if !os.is_dir(dataset_dir) {
			os.mkdir_all(dataset_dir)!
		}
		$if debug ? {
			println('Downloading ${data.file} from ${data.baseurl}${data.file}')
		}
		load_from_url(url: '${data.baseurl}${data.file}', target: target)!
	}
	uncompressed_path := os.join_path(dataset_dir, data.uncompressed_dir)
	if data.compressed && !os.is_dir(uncompressed_path) {
		$if debug ? {
			println('Extracting ${data.file}')
		}
		if is_tar {
			result := os.execute('tar -xvzf ${target} -C ${dataset_dir}')
			if result.exit_code != 0 {
				$if debug ? {
					println('Error extracting ${target}')
					println('Exit code: ${result.exit_code}')
					println('Output: ${result.output}')
				}
				return error_with_code('Error extracting ${target}', result.exit_code)
			}
		} else {
			file_content := os.read_file(target)!
			content := gzip.decompress(file_content.bytes(),
				verify_header_checksum: true
				verify_length: false
				verify_checksum: false
			)!
			os.write_file(uncompressed_path, content.bytestr())!
		}
	}

	return uncompressed_path
}
