module autograd

import vtl

fn test_exp() {
	f64_ctx := ctx[f64]()

	vals := []f64{len: 10, init: f64(index) / 10.0}

	x := f64_ctx.variable(vtl.from_1d(vals)!)

	mut f := x.exp()!

	f.backprop()!

	expected_vals := [1.0, 1.1051709180756477, 1.2214027581601699, 1.3498588075760032,
		1.4918246976412703, 1.6487212707001282, 1.8221188003905089, 2.0137527074704766,
		2.225540928492468, 2.45960311115695]
	expected_grad := vtl.from_1d(expected_vals)!

	assert x.grad.array_equal(expected_grad)
}
