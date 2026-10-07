module main

import vtl
import vtl.la

matrices := vtl.from_array([3, 0, 0, 0, 4, 0, 5, 0, 0, 0, 12, 0], [2, 2, 3])!
println('Frobenius: ${la.matrix_norm(matrices)!.to_array()}')
println('Nuclear: ${la.matrix_norm(matrices, ord: 'nuc')!.to_array()}')
println('Spectral: ${la.matrix_norm(matrices, ord: '2')!.to_array()}')
