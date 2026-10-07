import vtl
import vtl.la

fn main() {
	values := vtl.from_1d([1, 2, 3])!
	upper := la.diag(values, 1)!
	matrix := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	main_diagonal := la.diag(matrix, 0)!

	println('offset diagonal matrix:')
	println(upper)
	println('main diagonal: ${main_diagonal}')
}
