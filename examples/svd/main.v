module main

import vtl
import vtl.la

matrix := vtl.from_2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])!
u, singular_values, vt := la.svd(matrix, full_matrices: false)!
println('U (${u.shape}): ${u.to_array()}')
println('Singular values (${singular_values.shape}): ${singular_values.to_array()}')
println('V transpose (${vt.shape}): ${vt.to_array()}')
