module autograd

import vtl

fn test_pow() {
	ctx := new_ctx<f64>()

	x := ctx.variable(vtl.from_1d([3.0]) or { panic(@FN + ' failed') })
	y := ctx.variable(vtl.from_1d([2.0]) or { panic(@FN + ' failed') })

	mut f := pow(x, y) or { panic(@FN + ' failed') }

	f.backprop() or { panic(@FN + ' failed') }

	assert x.grad.equal(vtl.from_1d([6.0]) or { panic(@FN + ' failed') })
}
