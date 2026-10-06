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
		'features': vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [2, 2])!
		'targets':  vtl.from_1d[f64]([0.0, 1.0])!
	}
	npz.write(path, arrays)!
	features := npz.read[f64](path, 'features')!
	targets := npz.read[f64](path, 'targets')!
	println('Arrays: ${npz.members(path)!}')
	println('Features (${features.shape}): ${features.to_array()}')
	println('Targets (${targets.shape}): ${targets.to_array()}')
}
