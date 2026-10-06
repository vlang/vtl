module main

import vtl

fn main() {
	indices := vtl.from_2d([[0], [1]])!
	first := vtl.from_2d([[10, 20, 30]])!
	second := vtl.from_2d([[100], [200]])!
	selected := vtl.choose(indices, [first, second])!
	println('Selected values: ${selected.to_array()}')
}
