module main

import vtl
import vtl.la

batch := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3])!
println('Main diagonals: ${la.trace_axes(batch, axis1: 1, axis2: 2)!.to_array()}')
println('Offset diagonals: ${la.trace_axes(batch, axis1: 1, axis2: 2, offset: 1)!.to_array()}')
