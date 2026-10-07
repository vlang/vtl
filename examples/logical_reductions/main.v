import vtl

fn main() {
	values := vtl.from_array([0, 1, 2, 0, 3, 4], [2, 3]) or { panic(err) }
	println('all per row: ${values.all_axis(1, false) or { panic(err) }}')
	println('any per column: ${values.any_axis(0, true) or { panic(err) }}')
}
