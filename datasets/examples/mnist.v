module main

import vtl.datasets

fn main() {
	mut train_ds := datasets.load_mnist(.train, batch_size: 6) ?

	mut i := 0
	for {
		batch := train_ds.next() or { break }
		println('Labels Batch #${i++}:\n${*batch.labels}')
		println('')

		if i == 10 {
			break
		}
	}
}
