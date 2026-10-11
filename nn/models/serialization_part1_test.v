module models

import os
import json2
import vtl
import vtl.nn.types

const test_model_dir_1 = '/tmp/vtl_test_models_1'

fn setup_test_dir_1() string {
	os.mkdir_all(test_model_dir_1) or { panic(err) }
	return test_model_dir_1
}

fn cleanup_test_dir_1() {
	os.rmdir_all(test_model_dir_1) or {}
}

fn test_save_and_load_simple_model() {
	test_dir := setup_test_dir_1()
	defer {
		cleanup_test_dir_1()
	}

	// Create a simple model: input(1,2) -> linear(3) -> relu -> linear(2)
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.linear(3)
	nn.relu()
	nn.linear(2)

	// Save model
	model_path := '${test_dir}/simple_model.json'
	nn.save(model_path)!

	// Verify file was created
	assert os.exists(model_path)

	// Load the saved JSON and verify structure
	content := os.read_file(model_path)!
	model := json2.decode[ModelFile](content)!

	// Check version
	assert model.version == '1.0'

	// Check layer count (input + linear + relu + linear)
	assert model.layers.len == 4

	// Check layer types
	assert model.layers[0].layer_type == 'InputLayer'
	assert model.layers[1].layer_type == 'LinearLayer'
	assert model.layers[2].layer_type == 'ReLULayer'
	assert model.layers[3].layer_type == 'LinearLayer'

	// Check weights exist for linear layers
	assert 'weight' in model.layer_data[1].weights
	assert 'bias' in model.layer_data[1].weights
	assert 'weight' in model.layer_data[3].weights
	assert 'bias' in model.layer_data[3].weights

	// ReLU should have no weights
	assert model.layer_data[2].weights.len == 0
}

fn test_save_model_with_new_activation_layers() {
	test_dir := setup_test_dir_1()
	defer {
		cleanup_test_dir_1()
	}
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([3])
	nn.softplus()
	nn.selu()
	nn.hardswish()
	model_path := '${test_dir}/activation_model.json'
	nn.save(model_path)!
	content := os.read_file(model_path)!
	model := json2.decode[ModelFile](content)!
	assert model.layers.map(it.layer_type) == ['InputLayer', 'SoftplusLayer', 'SELULayer',
		'HardSwishLayer']
	assert model.layer_data[1].weights.len == 0
	assert model.layer_data[2].weights.len == 0
	assert model.layer_data[3].weights.len == 0

	mut restored := sequential_with_layers[f64]([]types.Layer[f64]{})
	restored.input([3])
	restored.softplus()
	restored.selu()
	restored.hardswish()
	restored.load_weights(model_path)!
	assert restored.info.layer_types == ['InputLayer', 'SoftplusLayer', 'SELULayer', 'HardSwishLayer']
}

fn test_save_with_loss_and_epoch() {
	test_dir := setup_test_dir_1()
	defer {
		cleanup_test_dir_1()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.linear(3)
	nn.relu()

	model_weights_path := '${test_dir}/model_with_metadata.json'
	nn.save_checkpoint(model_weights_path, 42, 0.123)!

	content := os.read_file(model_weights_path)!
	model := json2.decode[ModelFile](content)!

	assert model.metadata.epoch == 42
	assert model.metadata.loss == 0.123
}

fn test_load_weights() {
	test_dir := setup_test_dir_1()
	defer {
		cleanup_test_dir_1()
	}

	// Create and save model with known weights
	mut nn1 := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn1.input([1, 2])
	nn1.linear(3)
	nn1.relu()
	nn1.linear(2)

	// Set specific weights
	vars := nn1.info.layers[1].variables()
	mut ll_weights := vars[0].value
	mut ll_bias := vars[1].value
	for i in 0 .. ll_weights.size() {
		ll_weights.set_nth(i, vtl.cast[f64](1.5))
	}
	for i in 0 .. ll_bias.size() {
		ll_bias.set_nth(i, vtl.cast[f64](0.5))
	}
	path := '${test_dir}/weights_test.json'
	nn1.save(path)!

	// Create a new model with same architecture
	mut nn2 := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn2.input([1, 2])
	nn2.linear(3)
	nn2.relu()
	nn2.linear(2)

	// Load weights into the new model
	nn2.load_weights(path)!

	// Verify weights were loaded correctly
	vars1 := nn1.info.layers[1].variables()
	vars2 := nn2.info.layers[1].variables()

	for i in 0 .. vars1[0].value.size() {
		orig := f64(vars1[0].value.get_nth(i))
		loaded := f64(vars2[0].value.get_nth(i))
		assert loaded == orig, 'Weight mismatch at ${i}: expected ${orig}, got ${loaded}'
	}

	for i in 0 .. vars1[1].value.size() {
		orig := f64(vars1[1].value.get_nth(i))
		loaded := f64(vars2[1].value.get_nth(i))
		assert loaded == orig, 'Bias mismatch at ${i}: expected ${orig}, got ${loaded}'
	}
}
