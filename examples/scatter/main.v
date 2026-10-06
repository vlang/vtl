module main

import vtl

fn main() {
	indices := vtl.from_array([1, 1, 0, 0], [2, 2])!
	updates := vtl.from_array([10, 20, 30, 40], [2, 2])!

	mut accumulated := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
	accumulated.scatter_add(indices, updates, 1)!
	println('After scatter_add: ${accumulated.to_array()}')

	mut replaced := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
	replaced.put_along_axis(indices, updates, 1)!
	println('After put_along_axis: ${replaced.to_array()}')
}
