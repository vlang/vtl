import vtl

rows := vtl.from_1d([3, 6, 6])!
columns := vtl.from_1d([4, 5, 1])!
coordinates := [rows, columns]

c_order := vtl.ravel_multi_index(coordinates, [7, 6])!
f_order := vtl.ravel_multi_index(coordinates, [7, 6], order: .f)!
println('C-order flat indices: ${c_order.to_array()}')
println('F-order flat indices: ${f_order.to_array()}')

restored := vtl.unravel_index(c_order, [7, 6])!
println('restored rows: ${restored[0].to_array()}')
println('restored columns: ${restored[1].to_array()}')

wrapped := vtl.ravel_multi_index([vtl.from_1d([-1])!, vtl.from_1d([6])!], [4, 6],
	mode: .wrap
)!
println('wrapped index: ${wrapped.to_array()}')
