module main

import vtl

matrices := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3])!
lower := vtl.tril(matrices)!
upper := vtl.triu(matrices, k: 1)!
println('Batched lower triangles: ${lower.to_array()}')
println('Batched upper triangles: ${upper.to_array()}')
