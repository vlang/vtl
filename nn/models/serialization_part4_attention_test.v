module models

import os
import json2
import vtl.nn.types
import vtl.nn.layers

const test_model_dir_4b = '/tmp/vtl_test_models_4b'

fn setup_test_dir_4b() string {
	os.mkdir_all(test_model_dir_4b) or { panic(err) }
	return test_model_dir_4b
}

fn cleanup_test_dir_4b() {
	os.rmdir_all(test_model_dir_4b) or {}
}

fn test_multihead_attention_serialization() {
	test_dir := setup_test_dir_4b()
	defer {
		cleanup_test_dir_4b()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 10, 16])
	nn.multihead_attention(16, 4)

	path := '${test_dir}/attention_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json2.decode[ModelFile](content)!

	layer := model.layers[1]
	assert layer.layer_type == 'MultiHeadAttentionLayer'
	assert layer.config['embed_dim'] == 16
	assert layer.config['num_heads'] == 4

	// 4 weight matrices
	assert 'w_q' in model.layer_data[1].weights
	assert 'w_k' in model.layer_data[1].weights
	assert 'w_v' in model.layer_data[1].weights
	assert 'w_o' in model.layer_data[1].weights
}

fn test_layer_norm_serialization() {
	test_dir := setup_test_dir_4b()
	defer {
		cleanup_test_dir_4b()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 128])
	nn.layer_norm([128], layers.LayerNormConfig{
		eps: 1e-8
	})

	path := '${test_dir}/layernorm_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json2.decode[ModelFile](content)!

	assert model.layers[1].layer_type == 'LayerNormLayer'
	assert model.layers[1].config['normalized_shape_0'] == 128

	// gamma and beta should be present
	assert 'gamma' in model.layer_data[1].weights
	assert 'beta' in model.layer_data[1].weights
}
