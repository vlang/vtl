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

[params]
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

[params]
struct DatasetDownload {
	dataset    string
	baseurl    string
	extract    bool
	tar        bool
	urls_names map[string]string
}

fn download_dataset(data DatasetDownload) !map[string]string {
	mut loaded_paths := map[string]string{}

	for path, filename in data.urls_names {
		dataset_dir := get_cache_dir('datasets', data.dataset)

		if !os.is_dir(dataset_dir) {
			os.mkdir_all(dataset_dir)!
		}

		target := os.join_path(dataset_dir, filename)

		if os.exists(target) {
			$if debug ? {
				// we assume that the correct extraction process was done
				// before
				// TODO: check for extraction...
				println('${filename} already exists')
			}
		} else {
			$if debug ? {
				println('Downloading ${filename} from ${data.baseurl}${path}')
			}
			load_from_url(url: '${data.baseurl}${path}', target: target)!
			if data.extract {
				$if debug ? {
					println('Extracting ${target}')
				}
				if data.tar {
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
					umcompressed_filename := target#[0..-3]
					os.write_file(umcompressed_filename, content.bytestr())!
				}
			}
		}

		loaded_paths[path] = target
	}

	return loaded_paths
}
