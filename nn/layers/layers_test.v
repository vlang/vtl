module layers

import vtl
import vtl.autograd

fn ctx[T]() &autograd.Context[T] {
	return autograd.ctx[T]()
}

fn variable[T](c &autograd.Context[T], arr []T, shape []int) !&autograd.Variable[T] {
	t := vtl.from_array(arr, shape)!
	return c.variable(t)
}

// Linear layer: output shape and forward pass
fn test_linear_output_shape() {
	c := ctx[f64]()
	layer := linear_layer[f64](c, 4, 3)
	assert layer.output_shape() == [3], 'linear output_shape expected [3], got ${layer.output_shape()}'
}

fn test_linear_forward_shape() ! {
	c := ctx[f64]()
	layer := linear_layer[f64](c, 4, 3)
	input := variable[f64](c, [1.0, 0.0, -1.0, 0.5], [1, 4])!
	result := layer.forward(input)!
	// output should be [1, 3]
	assert result.value.shape == [1, 3], 'linear forward shape expected [1, 3], got ${result.value.shape}'
}

fn test_linear_variables_count() {
	c := ctx[f64]()
	layer := linear_layer[f64](c, 4, 3)
	vars := layer.variables()
	assert vars.len == 2, 'linear should have 2 variables (weights + bias), got ${vars.len}'
}

// BatchNorm layer: output shape
fn test_batchnorm_output_shape() {
	c := ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 8, BatchNorm1DConfig{})
	assert layer.output_shape() == [8], 'batchnorm output_shape expected [8], got ${layer.output_shape()}'
}

fn test_batchnorm_forward_shape() ! {
	c := ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 4, BatchNorm1DConfig{})
	input := variable[f64](c, [1.0, 2.0, 3.0, 4.0], [1, 4])!
	result := layer.forward(input)!
	assert result.value.shape == [1, 4], 'batchnorm forward shape expected [1, 4], got ${result.value.shape}'
}

fn test_batchnorm_variables_count() {
	c := ctx[f64]()
	layer := batchnorm1d_layer[f64](c, 4, BatchNorm1DConfig{})
	vars := layer.variables()
	assert vars.len == 2, 'batchnorm should have 2 variables (gamma + beta), got ${vars.len}'
}

// Embedding layer: output shape and variables
fn test_embedding_output_shape() {
	c := ctx[f64]()
	layer := embedding_layer[f64](c, 100, 16)
	assert layer.output_shape() == [16], 'embedding output_shape expected [16], got ${layer.output_shape()}'
}

fn test_embedding_variables_count() {
	c := ctx[f64]()
	layer := embedding_layer[f64](c, 100, 16)
	vars := layer.variables()
	assert vars.len == 1, 'embedding should have 1 variable (weight), got ${vars.len}'
	assert vars[0].value.shape == [100, 16], 'embedding weight shape expected [100, 16], got ${vars[0].value.shape}'
}
