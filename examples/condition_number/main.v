module main

import vtl
import vtl.la

matrices := vtl.from_array([1, 2, 3, 4, 2, 0, 0, 2], [2, 2, 2])!
spectral := la.cond(matrices, la.CondOptions{})!
one_norm := la.cond(matrices, la.CondOptions{
	ord: '1'
})!
println('2-norm condition numbers: ${spectral.to_array()}')
println('1-norm condition numbers: ${one_norm.to_array()}')
