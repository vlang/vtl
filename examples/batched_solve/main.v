module main

import vtl
import vtl.la

fn main() {
	matrices := vtl.from_array[f64]([2.0, 0, 0, 4, 1, 0, 0, 2], [2, 2, 2])!
	vector_rhs := vtl.from_array[f64]([2.0, 8, 3, 10], [2, 2])!
	vector_solutions := la.solve(matrices, vector_rhs)!
	println('Vector solutions (shape ${vector_solutions.shape}): ${vector_solutions.to_array()}')

	matrix_rhs := vtl.from_array[f64]([2.0, 4, 6, 8], [1, 2, 2])!
	matrix_solutions := la.solve(matrices, matrix_rhs)!
	println('Broadcast matrix solutions (shape ${matrix_solutions.shape}): ${matrix_solutions.to_array()}')
}
