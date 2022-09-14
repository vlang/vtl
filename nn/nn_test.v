module nn

import vtl.autograd
import vtl.nn.layers

fn test_nnc() {
	ctx := autograd.new_ctx<f64>()
	mut nncon := new_nnc<f64>(ctx)
	nncon.input([1, 2])
	nncon.sigmod()
	assert nncon.layers.len == 2
	assert layers.layer_output_shape<f64>(nncon.layers[0]) == [1, 2]
	assert layers.layer_output_shape<f64>(nncon.layers[1]) == [1, 2]
}

fn test_nn() {
	ctx := autograd.new_ctx<f64>()
	mut nn := new_nn<f64>(ctx)
}
