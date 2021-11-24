module main

import vtl.datasets

fn main() {
	mut ds := datasets.load_mnist(.test, batch_size: 32) ?
	println(ds)

	mut i := 0
	for {
		batch := ds.next() or { break }
		println('Batch number: ${i++}')
		// println(batch)
	}
}
