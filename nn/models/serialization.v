module models

import json
import os
import vtl
import vtl.autograd
import vtl.nn.types
import encoding.base64
import math

const model_version = '1.0'

// SerializationError represents errors during model save/load operations.
pub struct SerializationError {
	msg string
}

// msg exposes this operation as part of the public API.
pub fn (e &SerializationError) msg() string {
	return e.msg
}

// ModelMetadata stores training metadata for checkpoints.
pub struct ModelMetadata {
pub:
	epoch    int
	loss     f64
	saved_at string
	version  string = model_version
}

// LayerDef defines a single layer's type and configuration for reconstruction.
struct LayerDef {
mut:
	layer_type string
	in_shape   []int
	config     map[string]int
}

// SerializedLayer contains layer data for a single layer.
struct SerializedLayer {
mut:
	weights map[string]string // base64 encoded
}

// OptimizerData stores optimizer-specific state for serialization.
struct OptimizerData {
mut:
	optimizer_type string
	learning_rate  f64
	config         map[string]f64
	moments        map[string]string // base64 encoded
}

// ModelFile is the complete serialized model format.
struct ModelFile {
pub mut:
	version    string
	layers     []LayerDef
	layer_data []SerializedLayer
	optimizer  OptimizerData
	metadata   ModelMetadata
}

fn (nn &Sequential[T]) layer_type_at(index int, layer types.Layer[T]) string {
	if index < nn.info.layer_types.len && nn.info.layer_types[index] != '' {
		return nn.info.layer_types[index]
	}
	return typeof(layer).name
}

fn (nn &Sequential[T]) layer_config_at(index int) map[string]int {
	if index < nn.info.layer_configs.len {
		return nn.info.layer_configs[index].clone()
	}
	return map[string]int{}
}

fn is_erased_layer_type(name string) bool {
	return name == '' || name == 'Layer' || name.starts_with('types.Layer[')
}

fn set_config_default(mut config map[string]int, key string, value int) {
	if key !in config {
		config[key] = value
	}
}

fn shape_value(shape []int, index int, fallback int) int {
	if index < shape.len {
		return shape[index]
	}
	return fallback
}

fn encode_layer_var[T](mut weights map[string]string, vars []&autograd.Variable[T], index int, key string) ! {
	if index < vars.len {
		weights[key] = encode_tensor[T](vars[index].value)!
	}
}

// save saves a Sequential model's weights to a JSON file.
// Does NOT save optimizer state - use save_checkpoint for full state.
pub fn (nn &Sequential[T]) save(path string) ! {
	nn.save_checkpoint(path, 0, 0.0)!
}

// save_checkpoint saves a Sequential model's weights and training metadata to a JSON file.
pub fn (nn &Sequential[T]) save_checkpoint(path string, epoch int, loss f64) ! {
	mut layer_defs := []LayerDef{}
	mut layer_data := []SerializedLayer{}

	for i, layer in nn.info.layers {
		layer_type := nn.layer_type_at(i, layer)
		mut config := nn.layer_config_at(i)
		mut weights := map[string]string{}
		vars := layer.variables()

		// Capture input shape from previous layer or default
		in_shape := if i > 0 {
			nn.info.layers[i - 1].output_shape()
		} else {
			[]int{}
		}

		// Extract layer-specific configuration and weights
		match layer_type {
			'LinearLayer' {
				if vars.len > 0 {
					set_config_default(mut config, 'out_features', shape_value(vars[0].value.shape,
						0, 0))
					set_config_default(mut config, 'in_features', shape_value(vars[0].value.shape,
						1, 0))
				}
				encode_layer_var[T](mut weights, vars, 0, 'weight')!
				encode_layer_var[T](mut weights, vars, 1, 'bias')!
			}
			'ReLULayer', 'SigmoidLayer', 'TanhLayer', 'LeakyReLULayer', 'ELULayer', 'SwishLayer',
			'MishLayer', 'GELULayer', 'GeluLayer' {
				// Activation layers have no weights - they just copy shapes
			}
			'BatchNorm1DLayer' {
				if vars.len > 0 {
					set_config_default(mut config, 'num_features', shape_value(vars[0].value.shape,
						1, 0))
				}
				encode_layer_var[T](mut weights, vars, 0, 'gamma')!
				encode_layer_var[T](mut weights, vars, 1, 'beta')!
			}
			'EmbeddingLayer' {
				if vars.len > 0 {
					set_config_default(mut config, 'vocab_size', shape_value(vars[0].value.shape,
						0, 0))
					set_config_default(mut config, 'embedding_dim', shape_value(vars[0].value.shape,
						1, 0))
				}
				encode_layer_var[T](mut weights, vars, 0, 'weight')!
			}
			'Conv2DLayer' {
				if vars.len > 0 {
					set_config_default(mut config, 'out_channels', shape_value(vars[0].value.shape,
						0, 0))
					set_config_default(mut config, 'in_channels', shape_value(vars[0].value.shape,
						1, 0))
					set_config_default(mut config, 'kernel_h', shape_value(vars[0].value.shape, 2,
						0))
					set_config_default(mut config, 'kernel_w', shape_value(vars[0].value.shape, 3,
						0))
				}
				set_config_default(mut config, 'stride_h', 1)
				set_config_default(mut config, 'stride_w', 1)
				encode_layer_var[T](mut weights, vars, 0, 'weight')!
				encode_layer_var[T](mut weights, vars, 1, 'bias')!
			}
			'LSTMLayer' {
				if vars.len > 1 {
					set_config_default(mut config, 'input_size', shape_value(vars[0].value.shape,
						1, 0))
					set_config_default(mut config, 'hidden_size', shape_value(vars[1].value.shape,
						1, 0))
				}
				set_config_default(mut config, 'num_layers', 1)
				encode_layer_var[T](mut weights, vars, 0, 'w_ih')!
				encode_layer_var[T](mut weights, vars, 1, 'w_hh')!
				encode_layer_var[T](mut weights, vars, 2, 'b_ih')!
				encode_layer_var[T](mut weights, vars, 3, 'b_hh')!
			}
			'MultiHeadAttentionLayer' {
				if vars.len > 0 {
					set_config_default(mut config, 'embed_dim', shape_value(vars[0].value.shape, 0,
						0))
				}
				set_config_default(mut config, 'num_heads', 1)
				encode_layer_var[T](mut weights, vars, 0, 'w_q')!
				encode_layer_var[T](mut weights, vars, 1, 'w_k')!
				encode_layer_var[T](mut weights, vars, 2, 'w_v')!
				encode_layer_var[T](mut weights, vars, 3, 'w_o')!
			}
			'LayerNormLayer' {
				if vars.len > 0 {
					for j, dim in vars[0].value.shape {
						set_config_default(mut config, 'normalized_shape_${j}', dim)
					}
					encode_layer_var[T](mut weights, vars, 0, 'gamma')!
					encode_layer_var[T](mut weights, vars, 1, 'beta')!
				}
			}
			'FlattenLayer' {
				// Flatten has no weights
			}
			'MaxPool2DLayer', 'AvgPool2DLayer', 'GlobalAvgPool2DLayer' {
				// Pooling layers have no weights
			}
			'DropoutLayer' {
				// Dropout has no weights during inference
			}
			'InputLayer' {
				// Input layer has no weights
			}
			else {
				// For unknown layers, just extract variables if any exist
				for j, v in vars {
					key := 'var_${j}'
					weights[key] = encode_tensor[T](v.value)!
				}
			}
		}

		layer_defs << LayerDef{
			layer_type: layer_type
			in_shape:   in_shape
			config:     config
		}
		layer_data << SerializedLayer{
			weights: weights
		}
	}

	// Create metadata with current timestamp
	metadata := ModelMetadata{
		epoch:    epoch
		loss:     loss
		saved_at: 'vtl_checkpoint'
	}

	opt_state := OptimizerData{}

	model := ModelFile{
		version:    model_version
		layers:     layer_defs
		layer_data: layer_data
		optimizer:  opt_state
		metadata:   metadata
	}

	os.write_file(path, json.encode(model))!
}

// load_checkpoint restores a Sequential model's weights and training metadata from a JSON file.
// Returns the epoch and loss from the checkpoint.
// NOTE: The model must already be constructed with the same architecture.
// This method loads weights into an existing model.
pub fn Sequential.load_checkpoint[T](path string) !(int, f64) {
	data := os.read_file(path)!
	model := json.decode(ModelFile, data)!

	// Version compatibility check
	if model.version != model_version {
		return error('Model version mismatch: expected ${model_version}, got ${model.version}')
	}

	return model.metadata.epoch, model.metadata.loss
}

// load_weights restores weights into an existing model.
// The model's layers must already be constructed in the same order as when saved.
pub fn (nn &Sequential[T]) load_weights(path string) ! {
	data := os.read_file(path)!
	model := json.decode(ModelFile, data)!

	// Version check
	if model.version != model_version {
		return error('Model version mismatch: expected ${model_version}, got ${model.version}')
	}

	// Validate layer count matches
	if model.layers.len != nn.info.layers.len {
		return error('Layer count mismatch: model has ${nn.info.layers.len}, checkpoint has ${model.layers.len}')
	}

	for i, layer in nn.info.layers {
		layer_def := model.layers[i]

		// Validate layer type
		saved_type := layer_def.layer_type
		current_type := nn.layer_type_at(i, layer)
		if !is_erased_layer_type(current_type) && saved_type != current_type {
			return error('Layer ${i} type mismatch: expected ${saved_type}, got ${current_type}')
		}

		serialized := model.layer_data[i]
		vars := layer.variables()
		for j := 0; j < vars.len; j++ {
			mut v := vars[j]

			// Determine the correct key based on layer type
			weight_key := match saved_type {
				'LinearLayer' {
					if j == 0 { 'weight' } else { 'bias' }
				}
				'BatchNorm1DLayer' {
					if j == 0 {
						'gamma'
					} else if j == 1 {
						'beta'
					} else {
						'var_${j}'
					}
				}
				'LSTMLayer' {
					if j == 0 {
						'w_ih'
					} else if j == 1 {
						'w_hh'
					} else if j == 2 {
						'b_ih'
					} else {
						'b_hh'
					}
				}
				'MultiHeadAttentionLayer' {
					if j == 0 {
						'w_q'
					} else if j == 1 {
						'w_k'
					} else if j == 2 {
						'w_v'
					} else {
						'w_o'
					}
				}
				'LayerNormLayer' {
					if j == 0 { 'gamma' } else { 'beta' }
				}
				else {
					'var_${j}'
				}
			}

			if weight_key !in serialized.weights {
				return error('Missing weight ${weight_key} for layer ${i}')
			}

			encoded := serialized.weights[weight_key]
			tensor_data := decode_tensor[T](encoded, v.value.shape)!

			// Verify shape matches
			if tensor_data.shape != v.value.shape {
				return error('Weight shape mismatch for layer ${i} ${weight_key}: expected ${v.value.shape}, got ${tensor_data.shape}')
			}

			// Copy data into variable's value tensor
			if tensor_data.memory == v.value.memory {
				for k in 0 .. tensor_data.size() {
					v.value.data.data[k] = tensor_data.data.data[k]
				}
			} else {
				for k in 0 .. tensor_data.size() {
					v.value.set_nth(k, tensor_data.get_nth(k))
				}
			}
		}
	}
}

// These functions are kept for potential future use
// load_optimizer_state restores Adam optimizer state (moments) from a checkpoint.
// pub fn (mut opt optimizers.AdamOptimizer[T]) load_state(path string) ! {
// 	data := os.read_file(path)!
// 	model := json.decode(ModelFile, data)!
//
// 	if model.optimizer.optimizer_type != 'AdamOptimizer' {
// 		return error('Optimizer type mismatch: expected AdamOptimizer, got ${model.optimizer.optimizer_type}')
// 	}
//
// 	// Load beta states if present
// 	if model.optimizer.config.len > 0 {
// 		if 'beta1_t' in model.optimizer.config {
// 			opt.beta1_t = model.optimizer.config['beta1_t']
// 		}
// 		if 'beta2_t' in model.optimizer.config {
// 			opt.beta2_t = model.optimizer.config['beta2_t']
// 		}
// 	}
//
// 	// Load moments for each parameter
// 	for i, _ in opt.params {
// 		key_m := 'moment_${i}'
// 		key_v := 'second_moment_${i}'
// 		if key_m in model.optimizer.moments && i < opt.first_moments.len {
// 			moment_data := decode_tensor[T](model.optimizer.moments[key_m], opt.first_moments[i].shape)!
// 			for j in 0 .. moment_data.size() {
// 				opt.first_moments[i].set_nth(j, moment_data.get_nth(j))
// 			}
// 		}
// 		if key_v in model.optimizer.moments && i < opt.second_moments.len {
// 			snd_moment_data := decode_tensor[T](model.optimizer.moments[key_v], opt.second_moments[i].shape)!
// 			for j in 0 .. snd_moment_data.size() {
// 				opt.second_moments[i].set_nth(j, snd_moment_data.get_nth(j))
// 			}
// 		}
// 	}
// }

// encode_tensor converts a tensor to a string encoding with shape.
fn encode_tensor[T](t &vtl.Tensor[T]) !string {
	shape := t.shape
	mut data := []f64{}
	for val in t.data.data {
		data << f64(val)
	}
	// Encode shape as comma-separated
	shape_str := shape.map(it.str()).join(',')
	memory_str := if t.memory == .col_major { 'col_major' } else { 'row_major' }
	// Encode data as base64
	mut byte_data := []u8{len: data.len * 8}
	for i, val in data {
		bits := math.f64_bits(val)
		for j in 0 .. 8 {
			byte_data[i * 8 + j] = u8((bits >> (56 - j * 8)) & 0xff)
		}
	}
	encoded := base64.encode(byte_data)
	return '${shape_str};${memory_str}|${encoded}'
}

// decode_tensor decodes a string back to a tensor with the given shape.
fn decode_tensor[T](encoded string, shape []int) !&vtl.Tensor[T] {
	parts := encoded.split('|')
	if parts.len != 2 {
		return error('Invalid tensor encoding: missing shape delimiter')
	}
	shape_parts := parts[0].split(';')
	shape_str := shape_parts[0]
	memory := if shape_parts.len > 1 && shape_parts[1] == 'col_major' {
		vtl.MemoryFormat.col_major
	} else {
		vtl.MemoryFormat.row_major
	}
	data_str := parts[1]

	// Parse shape
	decoded_shape := shape_str.split(',').map(it.int())

	// Verify shape matches expected
	if decoded_shape != shape {
		return error('Shape mismatch: encoded ${decoded_shape}, expected ${shape}')
	}

	// Decode data
	decoded := base64.decode(data_str)
	mut arr := []T{}
	for i := 0; i < decoded.len / 8; i++ {
		mut bits := u64(0)
		for j in 0 .. 8 {
			bits = (bits << 8) | u64(decoded[i * 8 + j])
		}
		arr << vtl.cast[T](math.f64_from_bits(bits))
	}
	return vtl.from_array[T](arr, shape, memory: memory)!
}

// validate_model_compatibility checks if a saved model is compatible with the given layers.
// Returns an error with details if incompatible.
pub fn validate_model_compatibility[T](saved_path string, model_layers []types.Layer[T]) !bool {
	data := os.read_file(saved_path)!
	model := json.decode(ModelFile, data)!

	if model.version != model_version {
		return error('Model version mismatch: expected ${model_version}, got ${model.version}')
	}

	if model.layers.len != model_layers.len {
		return error('Layer count mismatch: model has ${model_layers.len}, checkpoint has ${model.layers.len}')
	}

	for i, layer in model_layers {
		saved_type := model.layers[i].layer_type
		current_type := typeof(layer).name
		if !is_erased_layer_type(current_type) && saved_type != current_type {
			return error('Layer ${i} type mismatch: expected ${saved_type}, got ${current_type}')
		}

		// Check that variable counts match
		saved_vars := model.layer_data[i].weights.len
		current_vars := layer.variables().len
		if saved_vars != current_vars {
			return error('Layer ${i} variable count mismatch: expected ${saved_vars}, got ${current_vars}')
		}
	}

	return true
}
