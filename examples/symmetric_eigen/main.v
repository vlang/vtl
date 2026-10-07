module main

import vtl
import vtl.la

matrices := vtl.from_array([1.0, 2, 2, 1, 4, 1, 1, 2], [2, 2, 2])!
values, vectors := la.eigh(matrices, la.EighOptions{})!
println('Eigenvalues: ${values.to_array()}')
println('Eigenvectors: ${vectors.to_array()}')
