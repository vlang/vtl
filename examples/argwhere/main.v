module main

import vtl

fn main() {
	values := vtl.from_2d([[0, 2, 0], [3, 4, 0]])!
	coordinates := vtl.argwhere[int](values)!
	println('Coordinates: ${coordinates.to_array()}')
	println('Shape: ${coordinates.shape}')
}
