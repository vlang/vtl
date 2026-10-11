module models

import os
import json2
import vtl.nn.types
import vtl.nn.layers

const test_model_dir_3 = '/tmp/vtl_test_models_3'

fn setup_test_dir_3() string {
	os.mkdir_all(test_model_dir_3) or { panic(err) }
	return test_model_dir_3
}

fn cleanup_test_dir_3() {
	os.rmdir_all(test_model_dir_3) or {}
}

fn test_gru_serialization() {
	test_dir := setup_test_dir_3()
	defer {
		cleanup_test_dir_3()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([3, 1, 2])
	nn.gru(2, 4)

	path := '${test_dir}/gru_test.json'
	nn.save(path)!
	content := os.read_file(path)!
	model := json2.decode[ModelFile](content)!

	gru_layer := model.layers[1]
	assert gru_layer.layer_type == 'GRULayer'
	assert gru_layer.config['input_size'] == 2
	assert gru_layer.config['hidden_size'] == 4
	assert model.layer_data[1].weights.len == 4
	assert 'w_ih' in model.layer_data[1].weights
	assert 'w_hh' in model.layer_data[1].weights
	assert 'b_ih' in model.layer_data[1].weights
	assert 'b_hh' in model.layer_data[1].weights

	mut restored := sequential_with_layers[f64]([]types.Layer[f64]{})
	restored.input([3, 1, 2])
	restored.gru(2, 4)
	restored.load_weights(path)!
	for i, variable in nn.info.layers[1].variables() {
		assert variable.value.to_array() == restored.info.layers[1].variables()[i].value.to_array()
	}
}

fn test_multilayer_perceptron_serialization() {
	test_dir := setup_test_dir_3()
	defer {
		cleanup_test_dir_3()
	}

	// MLP: input -> linear -> relu -> dropout -> linear -> softmax
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 784])
	nn.linear(256)
	nn.relu()
	nn.linear(10)

	path := '${test_dir}/mlp_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json2.decode[ModelFile](content)!

	assert model.layers.len == 4
	assert model.layers[0].layer_type == 'InputLayer'
	assert model.layers[1].layer_type == 'LinearLayer'
	assert model.layers[2].layer_type == 'ReLULayer'
	assert model.layers[3].layer_type == 'LinearLayer'
}

fn test_validate_model_compatibility() {
	test_dir := setup_test_dir_3()
	defer {
		cleanup_test_dir_3()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.linear(3)
	nn.relu()

	path := '${test_dir}/validate_test.json'
	nn.save(path)!

	// Valid compatibility check
	layers1 := nn.info.layers
	result := validate_model_compatibility[f64](path, layers1)!
	assert result == true

	// Invalid: wrong layer count
	mut nn2 := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn2.input([1, 2])
	nn2.linear(3)
	nn2.relu()
	nn2.linear(2)

	validate_model_compatibility[f64](path, nn2.info.layers) or {
		assert err.msg().contains('Layer count mismatch')
		return
	}
	assert false
}

fn test_model_serialization_rejects_missing_layer_data() ! {
	test_dir := setup_test_dir_3()
	defer {
		cleanup_test_dir_3()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([2])
	nn.linear(3)
	path := '${test_dir}/missing_layer_data.json'
	nn.save(path)!

	mut model := json2.decode[ModelFile](os.read_file(path)!)!
	model.layer_data.delete_last()
	os.write_file(path, json2.encode(model))!

	nn.load_weights(path) or {
		assert err.msg().contains('Layer data count mismatch')
		return
	}
	assert false, 'expected load_weights to reject missing layer data'
}

fn test_validate_model_compatibility_rejects_missing_layer_data() ! {
	test_dir := setup_test_dir_3()
	defer {
		cleanup_test_dir_3()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([2])
	nn.linear(3)
	path := '${test_dir}/compat_missing_layer_data.json'
	nn.save(path)!

	mut model := json2.decode[ModelFile](os.read_file(path)!)!
	model.layer_data.delete_last()
	os.write_file(path, json2.encode(model))!

	validate_model_compatibility[f64](path, nn.info.layers) or {
		assert err.msg().contains('Layer data count mismatch')
		return
	}
	assert false, 'expected validation to reject missing layer data'
}

fn test_load_checkpoint_returns_metadata() {
	test_dir := setup_test_dir_3()
	defer {
		cleanup_test_dir_3()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.linear(3)

	path := '${test_dir}/checkpoint_test.json'
	nn.save_checkpoint(path, 100, 0.00123)!

	epoch, loss := Sequential.load_checkpoint[f64](path)!
	assert epoch == 100
	assert loss == 0.00123
}

fn test_convolutional_network_serialization() {
	test_dir := setup_test_dir_3()
	defer {
		cleanup_test_dir_3()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 1, 28, 28])
	nn.conv2d(1, 8, [3, 3], layers.Conv2DConfig{
		padding: [1, 1]
		stride:  [1, 1]
	})
	nn.relu()
	nn.avgpool2d([2, 2], [0, 0], [2, 2])
	nn.flatten()
	nn.linear(32)
	nn.relu()
	nn.linear(10)

	path := '${test_dir}/convnet_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json2.decode[ModelFile](content)!

	assert model.layers[0].layer_type == 'InputLayer'
	assert model.layers[1].layer_type == 'Conv2DLayer'
	assert model.layers[2].layer_type == 'ReLULayer'
	assert model.layers[3].layer_type == 'AvgPool2DLayer'
	assert model.layers[4].layer_type == 'FlattenLayer'
	assert model.layers[5].layer_type == 'LinearLayer'
	assert model.layers[6].layer_type == 'ReLULayer'
	assert model.layers[7].layer_type == 'LinearLayer'

	// Conv2D should have weight and bias
	assert 'weight' in model.layer_data[1].weights
	assert 'bias' in model.layer_data[1].weights
}
