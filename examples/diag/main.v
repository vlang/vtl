import vtl
import vtl.la

fn main() {
	values := vtl.from_1d([1, 2, 3])!
	upper := la.diag(values, 1)!
	matrix := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	main_diagonal := la.diag(matrix, 0)!
	volume := vtl.from_array([]int{len: 24, init: index}, [2, 3, 4])!
	volume_diagonal := vtl.diagonal(volume, axis1: 0, axis2: 2)!

	println('offset diagonal matrix:')
	println(upper)
	println('main diagonal: ${main_diagonal}')
	println('N-D diagonal view shape: ${volume_diagonal.shape}')
	println('N-D diagonal values: ${volume_diagonal}')
}
