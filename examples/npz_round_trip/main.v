module main

import os
import vtl
import vtl.npz

fn main() {
	if os.args.len > 1 {
		path := os.args[1]
		weights := npz.read[f64](path, 'weights')!
		labels := npz.read[i32](path, 'labels')!
		mask := npz.read[bool](path, 'mask')!
		println('NumPy archive members: ${npz.members(path)!}')
		println('Weights (${weights.shape}): ${weights.to_array()}')
		println('Labels: ${labels.to_array()}')
		println('Mask: ${mask.to_array()}')
		return
	}
	path := os.join_path(os.temp_dir(), 'vtl_npz_example.npz')
	defer {
		os.rm(path) or {}
	}
	arrays := {
		'features': npz.array[f64](vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [2, 2])!)
		'labels':   npz.array[i32](vtl.from_1d[i32]([0, 1])!)
		'mask':     npz.array[bool](vtl.from_1d[bool]([true, false])!)
	}
	npz.write_arrays(path, arrays)!
	features := npz.read[f64](path, 'features')!
	labels := npz.read[i32](path, 'labels')!
	mask := npz.read[bool](path, 'mask')!
	println('Arrays: ${npz.members(path)!}')
	println('Features (${features.shape}): ${features.to_array()}')
	println('Labels: ${labels.to_array()}')
	println('Mask: ${mask.to_array()}')
}
