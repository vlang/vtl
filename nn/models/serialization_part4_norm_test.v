module models

import os
import json2
import vtl.nn.types
import vtl.nn.layers

const test_model_dir_4c = '/tmp/vtl_test_models_4c'

fn setup_test_dir_4c() string {
	os.mkdir_all(test_model_dir_4c) or { panic(err) }
	return test_model_dir_4c
}

fn cleanup_test_dir_4c() {
	os.rmdir_all(test_model_dir_4c) or {}
}

fn test_group_norm_serialization() {
	test_dir := setup_test_dir_4c()
	defer {
		cleanup_test_dir_4c()
	}
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([4, 8, 8])
	nn.group_norm(2, layers.GroupNormConfig{})
	path := '${test_dir}/groupnorm_test.json'
	nn.save(path)!
	content := os.read_file(path)!
	model := json2.decode[ModelFile](content)!
	assert model.layers[1].layer_type == 'GroupNormLayer'
	assert model.layers[1].config['num_groups'] == 2
	assert 'gamma' in model.layer_data[1].weights
	assert 'beta' in model.layer_data[1].weights
}

fn test_rms_norm_serialization() {
	test_dir := setup_test_dir_4c()
	defer {
		cleanup_test_dir_4c()
	}
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([8])
	nn.rms_norm([8], layers.RMSNormConfig{})
	path := '${test_dir}/rmsnorm_test.json'
	nn.save(path)!
	content := os.read_file(path)!
	model := json2.decode[ModelFile](content)!
	assert model.layers[1].layer_type == 'RMSNormLayer'
	assert model.layers[1].config['normalized_shape_0'] == 8
	assert 'weight' in model.layer_data[1].weights
	mut weight := nn.info.layers[1].variables()[0].value
	weight.set_nth(0, 3.0)
	nn.load_weights(path)!
	assert nn.info.layers[1].variables()[0].value.get_nth(0) == 1.0
}
