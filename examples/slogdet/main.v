module main

import vtl
import vtl.la

matrices := vtl.from_array([1, 2, 3, 4, 1, 2, 2, 4], [2, 2, 2])!
signs, logabsdets := la.slogdet(matrices)!
println('Signs: ${signs.to_array()}')
println('Log absolute determinants: ${logabsdets.to_array()}')
