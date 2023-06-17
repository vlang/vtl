module models

import vtl.nn.layers
import vtl.nn.types

const empty_layers = []types.Layer[f64]{}

fn test_nnc() {
	mut nn := sequential[f64]()
	nn.input([1, 2])
	nn.sigmod()
	assert nn.info.layers.len == 2
	assert nn.info.layers[0].output_shape() == [1, 2]
	assert nn.info.layers[1].output_shape() == [1, 2]
}

fn test_nn() {
	mut nn := sequential[f64]()
}
