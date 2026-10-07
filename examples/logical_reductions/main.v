import vtl

fn main() {
	values := vtl.from_array([0, 1, 2, 0, 3, 4], [2, 3]) or { panic(err) }
	all_rows := values.all_axis(1, false) or { panic(err) }
	any_columns := values.any_axis(0, true) or { panic(err) }
	println('all per row: ${all_rows.to_array()}')
	println('any per column: ${any_columns.to_array()} with shape ${any_columns.shape}')
	volume := vtl.from_array([0, 1, 2, 3, 0, 4, 5, 6], [2, 2, 2]) or { panic(err) }
	all_outer_axes := volume.all_axes([0, -1], false) or { panic(err) }
	println('all across outer axes: ${all_outer_axes.to_array()}')
}
