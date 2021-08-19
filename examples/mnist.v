module main

import vtl.datasets

fn main() {
	mut mnist_loader := datasets.load_mnist(.train, 32) or { panic(err) }

	for batch in mnist_loader {
		println(batch.str())
	}
}
