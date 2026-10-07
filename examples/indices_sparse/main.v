module main

import vtl

coordinates := vtl.indices_sparse([2, 3, 4])!
for axis, coordinate in coordinates {
	println('axis ${axis}: shape=${coordinate.shape}, values=${coordinate.to_array()}')
}
