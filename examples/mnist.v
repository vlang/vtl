module main

import vtl.datasets

fn main() {
	mut mnist_loader := datasets.load_mnist(.train, 32) or { panic(err) }
	println(mnist_loader)

	for batch in mnist_loader {
		println(batch.str())
	}
}
