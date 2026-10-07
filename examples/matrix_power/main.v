module main

import vtl
import vtl.la

matrices := vtl.from_array([0, 1, -1, 0, 2, 0, 0, 3], [2, 2, 2])!
println('Squared: ${la.matrix_power(matrices, 2)!.to_array()}')
println('Inverse: ${la.matrix_power(matrices, -1)!.to_array()}')
