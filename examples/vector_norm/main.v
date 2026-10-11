module main

import vtl
import vtl.la

fn main() {
	values := vtl.from_1d([3.0, -4.0, 0.0])!
	println('L1 norm: ${la.vector_norm(values, 1)!.get_nth(0)}')
	println('L2 norm: ${la.vector_norm(values, 2)!.get_nth(0)}')
	println('Nonzero count: ${la.vector_norm(values, 0)!.get_nth(0)}')

	matrix := vtl.from_2d([[3.0, 4.0], [0.0, 12.0]])!
	println('Row L2 norms: ${la.vector_norm_axis(matrix, 2, 1)!.to_array()}')

	tensor := vtl.from_array([f64(1), 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	tuple_axis := la.vector_norm_axes(tensor, 2, [0, 2], false)!
	println('L2 norms over axes [0, 2]: ${tuple_axis.to_array()}')
}
