module main

import math
import vtl
import vtl.autograd

fn main() {
	mut ctx := autograd.ctx[f64]()
	x := ctx.variable(vtl.from_1d([2.0, math.nan(), math.inf(1), math.inf(-1)])!)
	mut cleaned := x.nan_to_num(nan: 0.0, posinf: 10.0, neginf: -10.0)!
	println('cleaned values: ${cleaned.value.to_array()}')
	mut loss := cleaned.sum()!
	loss.backprop()!
	println('input gradient: ${x.grad.to_array()}')
}
