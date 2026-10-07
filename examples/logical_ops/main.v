import vtl

fn main() {
	values := vtl.from_array([0, 2, -3, 0], [2, 2])!
	condition := vtl.from_1d([1, 0])!
	selected := values.logical_and(condition)!
	assert selected.to_array() == [false, false, true, false]
	println('Logical mask: ${selected.to_array()}')
}
