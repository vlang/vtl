module autograd

import vtl
import vtl.autograd

fn test_pow() {
	ctx := new_ctx<f64>()

	x := ctx.variable(vtl.from_1d([3.0]))
	y := ctx.variable(vtl.from_1d([2.0]))

	mut f := pow(x, y)

	f.backprop()

	assert x.grad.equal(vtl.from_1d([6.0]))
}
