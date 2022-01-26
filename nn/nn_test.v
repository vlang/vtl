module nn

import vtl.autograd

fn test_nn() {
	mut ctx := autograd.new_ctx<f64>()
	mut nncon := new_nn_container<f64>(ctx)
	nncon.input([1, 2])
}
