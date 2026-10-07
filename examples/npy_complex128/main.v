module main

import math.complex as cmplx
import os
import vtl
import vtl.npy

fn main() {
	path := os.join_path(os.temp_dir(), 'vtl_complex128_example.npy')
	defer {
		os.rm(path) or {}
	}

	values := vtl.from_1d([
		cmplx.complex(1.5, -2.25),
		cmplx.complex(-3.0, 4.75),
	])!
	npy.write(path, values)!
	restored := npy.read[cmplx.Complex](path)!

	println('dtype: ${restored.dtype()}')
	println('values: ${restored.to_array()}')
}
