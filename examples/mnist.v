module main

import vtl.datasets

fn main() {
	mut ds := datasets.load_mnist(.test, 32) or { panic(err) }
	println(ds)

	mut i := 0
	for {
		batch := ds.next() or { break }
		println('Batch number: ${i++}')
		// println(batch.str())
	}
}
