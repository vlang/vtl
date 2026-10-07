module main

import vtl
import vtl.la

single := vtl.from_2d([[1.0, 2.0], [2.0, 4.0]])!
batch := vtl.from_array([1.0, 0, 0, 1, 1, 1, 1, 2, 2, 4, 3, 6], [2, 3, 2])!
println('Single matrix rank: ${la.matrix_rank(single, 0)!}')
println('Batch matrix ranks: ${la.matrix_rank_batch(batch, 0)!.to_array()}')
