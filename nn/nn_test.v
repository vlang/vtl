module nn

import vtl.autograd
import vtl.nn.layers

fn test_nnc() {
	ctx := autograd.new_ctx<f64>()
	mut nn := new_nn<f64>(ctx)
	nn.input([1, 2])
	nn.sigmod()
	assert nn.info.layers.len == 2
	assert layers.layer_output_shape<f64>(nn.info.layers[0]) == [1, 2]
	assert layers.layer_output_shape<f64>(nn.info.layers[1]) == [1, 2]
}

fn test_nn() {
	ctx := autograd.new_ctx<f64>()
	mut nn := new_nn<f64>(ctx)
}
