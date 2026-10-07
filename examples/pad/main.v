module main

import vtl

fn main() {
	values := vtl.from_1d([1, 2, 3])!
	reflected := vtl.pad[int](values, [[2, 2]], .reflect, 0)!
	symmetric := vtl.pad[int](values, [[2, 2]], .symmetric, 0)!
	constant := vtl.pad[int](values, [[1, 1]], .constant, -1)!

	println('reflect: ${reflected.to_array()}')
	println('symmetric: ${symmetric.to_array()}')
	println('constant: ${constant.to_array()}')
	assert reflected.to_array() == [3, 2, 1, 2, 3, 2, 1]
	assert symmetric.to_array() == [2, 1, 1, 2, 3, 3, 2]
	assert constant.to_array() == [-1, 1, 2, 3, -1]
}
