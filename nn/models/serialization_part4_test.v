module models

import os
import json2
import math
import vtl
import vtl.nn.types
import vtl.nn.layers

const test_model_dir_4 = '/tmp/vtl_test_models_4'

fn setup_test_dir_4() string {
	os.mkdir_all(test_model_dir_4) or { panic(err) }
	return test_model_dir_4
}

fn cleanup_test_dir_4() {
	os.rmdir_all(test_model_dir_4) or {}
}

fn test_conv1d_serialization() {
	test_dir := setup_test_dir_4()
	defer {
		cleanup_test_dir_4()
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
	test_dir := setup_test_dir_4()
	defer {
		cleanup_test_dir_4()
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

fn test_multihead_attention_serialization() {
	test_dir := setup_test_dir_4()
	defer {
		cleanup_test_dir_4()
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
	test_dir := setup_test_dir_4()
	defer {
		cleanup_test_dir_4()
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

fn test_group_norm_serialization() {
	test_dir := setup_test_dir_4()
	defer {
		cleanup_test_dir_4()
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
	test_dir := setup_test_dir_4()
	defer {
		cleanup_test_dir_4()
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

fn test_round_trip_weights_allclose_1e9() {
	test_dir := setup_test_dir_4()
	defer {
		cleanup_test_dir_4()
	}

	mut nn1 := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn1.input([1, 4])
	nn1.linear(3)
	nn1.relu()
	nn1.linear(2)

	vars := nn1.info.layers[1].variables()
	mut w := vars[0].value
	mut b := vars[1].value
	for i in 0 .. w.size() {
		w.set_nth(i, vtl.cast[f64](f64(i) * 0.11 + 0.01))
	}
	for i in 0 .. b.size() {
		b.set_nth(i, vtl.cast[f64](f64(i) * 0.07))
	}

	path := '${test_dir}/roundtrip_allclose.json'
	nn1.save(path)!

	mut nn2 := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn2.input([1, 4])
	nn2.linear(3)
	nn2.relu()
	nn2.linear(2)
	nn2.load_weights(path)!

	assert weights_allclose(nn1, nn2, 1e-9), 'round-trip weights differ beyond 1e-9'
}

fn weights_allclose(a &Sequential[f64], b &Sequential[f64], tol f64) bool {
	if a.info.layers.len != b.info.layers.len {
		return false
	}
	for i, layer in a.info.layers {
		vars_a := layer.variables()
		vars_b := b.info.layers[i].variables()
		if vars_a.len != vars_b.len {
			return false
		}
		for j in 0 .. vars_a.len {
			va := vars_a[j].value
			vb := vars_b[j].value
			if va.size() != vb.size() {
				return false
			}
			for k in 0 .. va.size() {
				da := f64(va.get_nth(k))
				db := f64(vb.get_nth(k))
				if math.abs(da - db) > tol {
					return false
				}
			}
		}
	}
	return true
}
