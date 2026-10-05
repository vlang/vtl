module main

import vtl
import vtl.autograd

ctx := autograd.ctx[f64]()

x := ctx.variable(vtl.from_1d([3.0])!)
y := ctx.variable(vtl.from_1d([2.0])!)

println(x)
println(y)

mut pow_result := x.pow(y)!

pow_result.backprop()!

println(pow_result)
println(x.grad)
