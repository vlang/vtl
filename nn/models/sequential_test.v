module models

import vtl.nn.layers

fn test_nnc() {
	mut nn := sequential[f64]()
	nn.input([1, 2])
	nn.sigmod()
	assert nn.info.layers.len == 2
	assert layers.layer_output_shape[f64](nn.info.layers[0]) == [1, 2]
	assert layers.layer_output_shape[f64](nn.info.layers[1]) == [1, 2]
}

fn test_nn() {
	mut nn := sequential[f64]()
}
