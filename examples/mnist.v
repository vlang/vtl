module main

import vtl.datasets

fn main() {
	mut mnist_loader := datasets.load_mnist(.test, 32) or { panic(err) }
	println(mnist_loader)

	mut i := 0
	for batch in mnist_loader {
		println('Batch number: ${i++}')
		// println(batch.str())
	}
}
