module main

import vtl
import vtl.la

matrices := vtl.from_array([4.0, 7, 2, 6, 2, 0, 0, 4], [2, 2, 2])!
println('Determinants: ${la.det(matrices)!.to_array()}')
println('Inverses: ${la.inv(matrices)!.to_array()}')
