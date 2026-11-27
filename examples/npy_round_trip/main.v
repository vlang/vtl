import vtl
import vtl.npy
import os

fn main() {
	path := os.join_path(os.temp_dir(), 'vtl_matrix.npy')
	defer {
		os.rm(path) or {}
	}
	values := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [2, 2])!
	npy.write(path, values)!
	loaded := npy.read[f64](path)!
	println('shape: ${loaded.shape}')
	println('values: ${loaded.to_array()}')
	mut mapped := npy.open_mapped[f64](path)!
	defer {
		mapped.close()
	}
	first_row := mapped.read_rows(0, 1)!
	println('first mapped row: ${first_row.to_array()}')
}
