module models

import os
import json2
import vtl
import vtl.nn.types
import vtl.nn.layers

const test_model_dir_2 = '/tmp/vtl_test_models_2'

fn setup_test_dir_2() string {
	os.mkdir_all(test_model_dir_2) or { panic(err) }
	return test_model_dir_2
}

fn cleanup_test_dir_2() {
	os.rmdir_all(test_model_dir_2) or {}
}

fn test_version_mismatch() {
	test_dir := setup_test_dir_2()
	defer {
		cleanup_test_dir_2()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.linear(3)

	path := '${test_dir}/version_test.json'
	nn.save(path)!

	// Modify the version in the file
	content := os.read_file(path)!
	mut model := json2.decode[ModelFile](content)!
	model.version = '99.9'
	modified := json2.encode(model, escape_unicode: true, time_as_unix: true)
	os.write_file(path, modified)!

	// Try to load - should fail version check
	mut nn2 := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn2.input([1, 2])
	nn2.linear(3)

	nn2.load_weights(path) or {
		assert err.msg().contains('version mismatch')
		return
	}
	assert false
}

fn test_layer_count_mismatch() {
	test_dir := setup_test_dir_2()
	defer {
		cleanup_test_dir_2()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.linear(3)
	nn.relu()

	path := '${test_dir}/layer_count_test.json'
	nn.save(path)!

	// Try to load into model with different layer count
	mut nn2 := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn2.input([1, 2])
	nn2.linear(3)
	nn2.relu()
	nn2.linear(2)

	nn2.load_weights(path) or {
		assert err.msg().contains('Layer count mismatch')
		return
	}
	assert false
}

fn test_layer_type_mismatch() {
	test_dir := setup_test_dir_2()
	defer {
		cleanup_test_dir_2()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.linear(3)
	nn.relu()

	path := '${test_dir}/layer_type_test.json'
	nn.save(path)!

	// Modify file to change a layer type
	content := os.read_file(path)!
	mut model := json2.decode[ModelFile](content)!
	model.layers[1].layer_type = 'ReLULayer' // Changed from LinearLayer
	modified := json2.encode(model, escape_unicode: true, time_as_unix: true)
	os.write_file(path, modified)!

	mut nn2 := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn2.input([1, 2])
	nn2.linear(3)
	nn2.relu()

	nn2.load_weights(path) or {
		assert err.msg().contains('type mismatch')
		return
	}
	assert false
}

fn test_linear_layer_weights_serialization() {
	test_dir := setup_test_dir_2()
	defer {
		cleanup_test_dir_2()
	}

	// Create model with specific weight values
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([3])
	nn.linear(4)

	// Set known weights: shape [4, 3]
	weights_data := [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
	vars := nn.info.layers[1].variables()
	mut ll_weights := vars[0].value
	for i in 0 .. weights_data.len {
		ll_weights.set_nth(i, vtl.cast[f64](weights_data[i]))
	}
	path := '${test_dir}/linear_weights_test.json'
	nn.save(path)!

	// Load and verify
	mut nn2 := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn2.input([3])
	nn2.linear(4)
	nn2.load_weights(path)!

	vars2 := nn2.info.layers[1].variables()
	for i in 0 .. weights_data.len {
		loaded := f64(vars2[0].value.get_nth(i))
		assert loaded == weights_data[i], 'Weight ${i} mismatch: expected ${weights_data[i]}, got ${loaded}'
	}
}

fn test_batchnorm_serialization() {
	test_dir := setup_test_dir_2()
	defer {
		cleanup_test_dir_2()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 8])
	nn.batchnorm1d(8, layers.BatchNorm1DConfig{
		eps:      1e-5
		momentum: 0.1
	})

	// Set known values
	vars := nn.info.layers[1].variables()
	mut gamma := vars[0].value
	mut beta := vars[1].value
	for i in 0 .. gamma.size() {
		gamma.set_nth(i, vtl.cast[f64](f64(i) * 0.5))
		beta.set_nth(i, vtl.cast[f64](f64(i) * 0.1))
	}
	path := '${test_dir}/batchnorm_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json2.decode[ModelFile](content)!

	// BatchNorm1D should have gamma, beta, running_mean, running_var
	assert model.layer_data[1].weights.len >= 2
	assert 'gamma' in model.layer_data[1].weights
	assert 'beta' in model.layer_data[1].weights
}

fn test_lstm_serialization() {
	test_dir := setup_test_dir_2()
	defer {
		cleanup_test_dir_2()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 10, 20])
	nn.lstm(20, 15, 1)

	path := '${test_dir}/lstm_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json2.decode[ModelFile](content)!

	// Check LSTM layer config
	lstm_layer := model.layers[1]
	assert lstm_layer.layer_type == 'LSTMLayer'
	assert lstm_layer.config['input_size'] == 20
	assert lstm_layer.config['hidden_size'] == 15
	assert lstm_layer.config['num_layers'] == 1

	// LSTM should have 4 weight tensors
	assert 'w_ih' in model.layer_data[1].weights
	assert 'w_hh' in model.layer_data[1].weights
	assert 'b_ih' in model.layer_data[1].weights
	assert 'b_hh' in model.layer_data[1].weights
}
