module main

import os
import vtl.npz

fn main() {
	if os.args.len < 2 {
		eprintln('usage: v run ./vtl/examples/npz_read_compressed/main.v <archive.npz>')
		exit(1)
	}
	path := os.args[1]
	weights := npz.read[f64](path, 'weights')!
	println('Members: ${npz.members(path)!}')
	println('Weights (${weights.shape}): ${weights.to_array()}')
}
