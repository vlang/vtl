module models

import json
import os
import vtl
import vtl.autograd
import vtl.nn.types
import vtl.nn.layers
import encoding.base64

const model_version = '1.0'

// SerializationError represents errors during model save/load operations.
pub struct SerializationError {
	msg string
}

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
pub:
	version    string
	layers     []LayerDef
	layer_data []SerializedLayer
	optimizer  OptimizerData
	metadata   ModelMetadata
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
		layer_type := typeof(layer).name
		mut config := map[string]int{}
		mut weights := map[string]string{}

		// Capture input shape from previous layer or default
		in_shape := if i > 0 {
			nn.info.layers[i - 1].output_shape()
		} else {
			[]int{}
		}

		// Extract layer-specific configuration and weights
		match layer_type {
			'LinearLayer' {
				unsafe {
					ll := &layers.LinearLayer[T](layer)
					config['in_features'] = ll.weights.value.shape[1]
					config['out_features'] = ll.weights.value.shape[0]
					weights['weight'] = encode_tensor[T](ll.weights.value)!
					weights['bias'] = encode_tensor[T](ll.bias.value)!
				}
			}
			'ReLULayer', 'SigmoidLayer', 'TanhLayer', 'LeakyReLULayer', 'ELULayer', 'SwishLayer', 'MishLayer', 'GeluLayer' {
				// Activation layers have no weights - they just copy shapes
			}
			'BatchNorm1DLayer' {
				unsafe {
					bn := &layers.BatchNorm1DLayer[T](layer)
					config['num_features'] = bn.gamma.value.shape[1]
					weights['gamma'] = encode_tensor[T](bn.gamma.value)!
					weights['beta'] = encode_tensor[T](bn.beta.value)!
					weights['running_mean'] = encode_tensor[T](bn.running_mean)!
					weights['running_var'] = encode_tensor[T](bn.running_var)!
				}
			}
			'EmbeddingLayer' {
				unsafe {
					em := &layers.EmbeddingLayer[T](layer)
					config['vocab_size'] = em.weight.value.shape[0]
					config['embedding_dim'] = em.weight.value.shape[1]
					weights['weight'] = encode_tensor[T](em.weight.value)!
				}
			}
			'Conv2DLayer' {
				unsafe {
					cv := &layers.Conv2DLayer[T](layer)
					config['in_channels'] = cv.weight.value.shape[1]
					config['out_channels'] = cv.weight.value.shape[0]
					config['kernel_h'] = cv.weight.value.shape[2]
					config['kernel_w'] = cv.weight.value.shape[3]
					config['stride_h'] = cv.config.stride[0]
					config['stride_w'] = cv.config.stride[1]
					weights['weight'] = encode_tensor[T](cv.weight.value)!
					weights['bias'] = encode_tensor[T](cv.bias.value)!
				}
			}
			'LSTMLayer' {
				unsafe {
					lstm := &layers.LSTMLayer[T](layer)
					config['input_size'] = lstm.w_ih.value.shape[1]
					config['hidden_size'] = lstm.hidden_size
					config['num_layers'] = lstm.num_layers
					weights['w_ih'] = encode_tensor[T](lstm.w_ih.value)!
					weights['w_hh'] = encode_tensor[T](lstm.w_hh.value)!
					weights['b_ih'] = encode_tensor[T](lstm.b_ih.value)!
					weights['b_hh'] = encode_tensor[T](lstm.b_hh.value)!
				}
			}
			'MultiHeadAttentionLayer' {
				unsafe {
					mha := &layers.MultiHeadAttentionLayer[T](layer)
					config['embed_dim'] = mha.embed_dim
					config['num_heads'] = mha.num_heads
					weights['w_q'] = encode_tensor[T](mha.w_q.value)!
					weights['w_k'] = encode_tensor[T](mha.w_k.value)!
					weights['w_v'] = encode_tensor[T](mha.w_v.value)!
					weights['w_o'] = encode_tensor[T](mha.w_o.value)!
				}
			}
			'LayerNormLayer' {
				unsafe {
					ln := &layers.LayerNormLayer[T](layer)
					config['normalized_shape_0'] = ln.normalized_shape[0]
					if ln.normalized_shape.len > 1 {
						config['normalized_shape_1'] = ln.normalized_shape[1]
					}
				}
				// gamma/beta check is outside unsafe because ln.gamma is a pointer
				if layer.variables().len > 0 {
					unsafe {
						ln2 := &layers.LayerNormLayer[T](layer)
						weights['gamma'] = encode_tensor[T](ln2.gamma.value)!
						weights['beta'] = encode_tensor[T](ln2.beta.value)!
					}
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
				for j, v in layer.variables() {
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
		layer_data << SerializedLayer{weights: weights}
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

	mut var_idx := 0
	for i, layer in nn.info.layers {
		layer_def := model.layers[i]

		// Validate layer type
		saved_type := layer_def.layer_type
		current_type := typeof(layer).name
		if saved_type != current_type {
			return error('Layer ${i} type mismatch: expected ${saved_type}, got ${current_type}')
		}

		serialized := model.layer_data[i]
		for j, v in layer.variables() {
			key := 'var_${j}'

			// Determine the correct key based on layer type
			weight_key := match saved_type {
				'LinearLayer' {
					if j == 0 { 'weight' } else { 'bias' }
				}
				'BatchNorm1DLayer' {
					if j == 0 { 'gamma' } else if j == 1 { 'beta' } else { key }
				}
				'LSTMLayer' {
					if j == 0 { 'w_ih' } else if j == 1 { 'w_hh' } else if j == 2 { 'b_ih' } else { 'b_hh' }
				}
				'MultiHeadAttentionLayer' {
					if j == 0 { 'w_q' } else if j == 1 { 'w_k' } else if j == 2 { 'w_v' } else { 'w_o' }
				}
				'LayerNormLayer' {
					if j == 0 { 'gamma' } else { 'beta' }
				}
				else { key }
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
			for k in 0 .. tensor_data.size() {
				v.value.set_nth(k, tensor_data.get_nth(k))
			}
			var_idx++
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

// encode_tensor converts a tensor to a base64 encoded string with shape.
fn encode_tensor[T](t &vtl.Tensor[T]) !string {
	shape := t.shape
	mut data := []f64{}
	for i in 0 .. t.size() {
		data << f64(t.get_nth(i))
	}
	encoded := base64.encode(data)
	// Store shape alongside data
	shape_str := shape.join(',')
	return '${shape_str}|${encoded}'
}

// decode_tensor decodes a base64 string back to a tensor with the given shape.
fn decode_tensor[T](encoded string, shape []int) !&vtl.Tensor[T] {
	parts := encoded.splitn('|', 2)
	if parts.len != 2 {
		return error('Invalid tensor encoding: missing shape delimiter')
	}
	shape_str := parts[0]
	data_str := parts[1]

	// Parse shape
	decoded_shape := shape_str.split(',').map(it.int())

	// Verify shape matches expected
	if decoded_shape != shape {
		return error('Shape mismatch: encoded ${decoded_shape}, expected ${shape}')
	}

	// Decode data
	decoded := base64.decode(data_str)!
	mut arr := []T{}
	for i in 0 .. decoded.len {
		arr << vtl.cast[T](f64(decoded[i]))
	}
	return vtl.from_array[T](arr, shape)!
}

// validate_model_compatibility checks if a saved model is compatible with the given layers.
// Returns an error with details if incompatible.
pub fn validate_model_compatibility[T](saved_path string, layers []types.Layer[T]) !bool {
	data := os.read_file(saved_path)!
	model := json.decode(ModelFile, data)!

	if model.version != model_version {
		return error('Model version mismatch: expected ${model_version}, got ${model.version}')
	}

	if model.layers.len != layers.len {
		return error('Layer count mismatch: model has ${layers.len}, checkpoint has ${model.layers.len}')
	}

	for i, layer in layers {
		saved_type := model.layers[i].layer_type
		current_type := typeof(layer).name
		if saved_type != current_type {
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
