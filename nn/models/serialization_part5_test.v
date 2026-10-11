module models

import os
import json2
import vtl.nn.types

const test_model_dir_5 = '/tmp/vtl_test_models_5'

fn setup_test_dir_5() string {
	os.mkdir_all(test_model_dir_5) or { panic(err) }
	return test_model_dir_5
}

fn cleanup_test_dir_5() {
	os.rmdir_all(test_model_dir_5) or {}
}

fn test_readme_example() {
	test_dir := setup_test_dir_5()
	defer {
		cleanup_test_dir_5()
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
	model := json2.decode[ModelFile](content)!

	assert model.layers.len == 4
	assert model.metadata.version == '1.0'

	// Weights encoded as base64 with shape info
	assert model.layer_data[1].weights['weight'].len > 100
	assert model.layer_data[1].weights['bias'].len > 10
}
