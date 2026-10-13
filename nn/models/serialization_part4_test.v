module models

import os
import json2
import vtl.nn.types
import vtl.nn.layers

const test_model_dir_4a = '/tmp/vtl_test_models_4a'

fn setup_test_dir_4a() string {
	os.mkdir_all(test_model_dir_4a) or { panic(err) }
	return test_model_dir_4a
}

fn cleanup_test_dir_4a() {
	os.rmdir_all(test_model_dir_4a) or {}
}

fn test_conv1d_serialization() {
	test_dir := setup_test_dir_4a()
	defer {
		cleanup_test_dir_4a()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([2, 12])
	nn.conv1d(4, 3, layers.Conv1DConfig{
		stride:  2
		padding: 1
		groups:  2
	})
	path := '${test_dir}/conv1d_test.json'
	nn.save(path)!
	model := json2.decode[ModelFile](os.read_file(path)!)!

	assert model.layers[1].layer_type == 'Conv1DLayer'
	assert model.layers[1].config['in_channels'] == 2
	assert model.layers[1].config['out_channels'] == 4
	assert model.layers[1].config['kernel_size'] == 3
	assert model.layers[1].config['stride'] == 2
	assert model.layers[1].config['padding'] == 1
	assert model.layers[1].config['groups'] == 2
	assert 'var_0' in model.layer_data[1].weights
	assert 'var_1' in model.layer_data[1].weights
}

fn test_embedding_layer_serialization() {
	test_dir := setup_test_dir_4a()
	defer {
		cleanup_test_dir_4a()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 10])
	nn.embedding(16, 8)

	path := '${test_dir}/embedding_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json2.decode[ModelFile](content)!

	assert model.layers[1].layer_type == 'EmbeddingLayer'
	assert model.layers[1].config['vocab_size'] == 16
	assert model.layers[1].config['embedding_dim'] == 8
	assert 'weight' in model.layer_data[1].weights
}
