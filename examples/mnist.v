module main

import vtl.datasets { load_mnist }

fn main() {
	load_mnist() or { panic(err) }
}
