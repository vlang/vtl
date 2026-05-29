module models

import os
import json
import vtl
import vtl.nn.types
import vtl.nn.layers

const test_model_dir = '/tmp/vtl_test_models'

fn setup_test_dir() string {
	os.mkdir_all(test_model_dir) or { panic(err) }
	return test_model_dir
}

fn cleanup_test_dir() {
	os.rmdir_all(test_model_dir) or {}
}

fn test_save_and_load_simple_model() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
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
	model := json.decode(ModelFile, content)!

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

fn test_save_with_loss_and_epoch() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.linear(3)
	nn.relu()

	model_weights_path := '${test_dir}/model_with_metadata.json'
	nn.save_checkpoint(model_weights_path, 42, 0.123)!

	content := os.read_file(model_weights_path)!
	model := json.decode(ModelFile, content)!

	assert model.metadata.epoch == 42
	assert model.metadata.loss == 0.123
}

fn test_load_weights() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
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

fn test_version_mismatch() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.linear(3)

	path := '${test_dir}/version_test.json'
	nn.save(path)!

	// Modify the version in the file
	content := os.read_file(path)!
	mut model := json.decode(ModelFile, content)!
	model.version = '99.9'
	modified := json.encode(model)
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
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
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
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.linear(3)
	nn.relu()

	path := '${test_dir}/layer_type_test.json'
	nn.save(path)!

	// Modify file to change a layer type
	content := os.read_file(path)!
	mut model := json.decode(ModelFile, content)!
	model.layers[1].layer_type = 'ReLULayer' // Changed from LinearLayer
	modified := json.encode(model)
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
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
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
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
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
	model := json.decode(ModelFile, content)!

	// BatchNorm1D should have gamma, beta, running_mean, running_var
	assert model.layer_data[1].weights.len >= 2
	assert 'gamma' in model.layer_data[1].weights
	assert 'beta' in model.layer_data[1].weights
}

fn test_lstm_serialization() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 10, 20])
	nn.lstm(20, 15, 1)

	path := '${test_dir}/lstm_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json.decode(ModelFile, content)!

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

fn test_multilayer_perceptron_serialization() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
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
	model := json.decode(ModelFile, content)!

	assert model.layers.len == 4
	assert model.layers[0].layer_type == 'InputLayer'
	assert model.layers[1].layer_type == 'LinearLayer'
	assert model.layers[2].layer_type == 'ReLULayer'
	assert model.layers[3].layer_type == 'LinearLayer'
}

fn test_validate_model_compatibility() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
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

fn test_load_checkpoint_returns_metadata() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
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
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
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
	model := json.decode(ModelFile, content)!

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

fn test_embedding_layer_serialization() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 10])
	nn.embedding(16, 8)

	path := '${test_dir}/embedding_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json.decode(ModelFile, content)!

	assert model.layers[1].layer_type == 'EmbeddingLayer'
	assert model.layers[1].config['vocab_size'] == 16
	assert model.layers[1].config['embedding_dim'] == 8
	assert 'weight' in model.layer_data[1].weights
}

fn test_multihead_attention_serialization() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 10, 16])
	nn.multihead_attention(16, 4)

	path := '${test_dir}/attention_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json.decode(ModelFile, content)!

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
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
	}

	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 128])
	nn.layer_norm([128], layers.LayerNormConfig{
		eps: 1e-8
	})

	path := '${test_dir}/layernorm_test.json'
	nn.save(path)!

	content := os.read_file(path)!
	model := json.decode(ModelFile, content)!

	assert model.layers[1].layer_type == 'LayerNormLayer'
	assert model.layers[1].config['normalized_shape_0'] == 128

	// gamma and beta should be present
	assert 'gamma' in model.layer_data[1].weights
	assert 'beta' in model.layer_data[1].weights
}

fn test_readme_example() {
	test_dir := setup_test_dir()
	defer {
		cleanup_test_dir()
	}

	// Example from the task: MLP like in MNIST
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 784])
	nn.linear(256)
	nn.relu()
	nn.linear(10)

	// Save
	model_path := '${test_dir}/mnist_model.json'
	nn.save(model_path)!

	// Verify structure again
	content := os.read_file(model_path)!
	model := json.decode(ModelFile, content)!

	assert model.layers.len == 4
	assert model.metadata.version == '1.0'

	// Weights encoded as base64 with shape info
	assert model.layer_data[1].weights['weight'].len > 100
	assert model.layer_data[1].weights['bias'].len > 10
}
