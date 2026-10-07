module main

import vtl
import vtl.la

matrices := vtl.from_array([3, 0, 0, 0, 4, 0, 5, 0, 0, 0, 12, 0], [2, 2, 3])!
println('Singular values: ${la.svdvals(matrices)!.to_array()}')
