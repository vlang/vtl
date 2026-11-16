module models

import json2
import math
import os
import rand
import vtl

const inference_checkpoint_format = 'vtl.sequential.inference'
const inference_checkpoint_version = 1

struct InferenceCheckpointFile {
pub:
	format  string
	version int
	model   string
	dtype   string
	layers  []InferenceCheckpointLayer
}

struct InferenceCheckpointLayer {
pub:
	type_name    string
	config       map[string]int
	output_shape []int
	parameters   []InferenceCheckpointParameter
}

struct InferenceCheckpointParameter {
pub:
	name   string
	shape  []int
	dtype  string
	values []f64
}

fn inference_checkpoint_dtype[T]() !string {
	$if T is f32 {
		return 'float32'
	} $else $if T is f64 {
		return 'float64'
	} $else {
		return error('inference checkpoints support only f32 and f64 Sequential models')
	}
}

fn inference_checkpoint_parameter_names(layer_type string, count int) ![]string {
	match layer_type {
		'InputLayer', 'ReLULayer' {
			if count != 0 {
				return error('inference checkpoint: ${layer_type} must not have parameters')
			}
			return []string{}
		}
		'LinearLayer' {
			if count != 2 {
				return error('inference checkpoint: LinearLayer requires weight and bias parameters')
			}
			return ['weight', 'bias']
		}
		else {
			return error('inference checkpoint: unsupported Sequential layer ${layer_type}')
		}
	}
}

fn inference_checkpoint_validate_model[T](nn &Sequential[T], validate_parameter_values bool) !string {
	dtype := inference_checkpoint_dtype[T]()!
	if nn.info.ctx.device_session != unsafe { nil } || os.getenv('VTL_USE_CUDA') == '1'
		|| os.getenv('VTL_USE_VULKAN') == '1' {
		return error('inference checkpoint: only CPU Sequential models are supported')
	}
	if nn.info.layers.len == 0 || nn.info.layer_types.len != nn.info.layers.len
		|| nn.info.layer_configs.len != nn.info.layers.len {
		return error('inference checkpoint: model layer metadata is incomplete')
	}
	for i, layer in nn.info.layers {
		layer_type := nn.info.layer_types[i]
		if layer_type !in ['InputLayer', 'LinearLayer', 'ReLULayer'] {
			return error('inference checkpoint: unsupported Sequential layer ${layer_type} at index ${i}')
		}
		inference_checkpoint_parameter_names(layer_type, layer.variables().len)!
		shape := layer.output_shape()
		if shape.len == 0 || shape.any(it <= 0) {
			return error('inference checkpoint: invalid output shape at layer ${i}')
		}
		if validate_parameter_values {
			for variable in layer.variables() {
				for index in 0 .. variable.value.size() {
					value := f64(variable.value.get_nth(index))
					if math.is_nan(value) || math.is_inf(value, 0) {
						return error('inference checkpoint: non-finite parameter at layer ${i}')
					}
				}
			}
		}
	}
	return dtype
}

// save_inference_checkpoint stores CPU inference weights and the supported
// Sequential architecture in a versioned, data-only JSON format. It does not
// store optimizer state or training-resume metadata.
pub fn (nn &Sequential[T]) save_inference_checkpoint(path string) ! {
	dtype := inference_checkpoint_validate_model[T](nn, true)!
	mut serialized_layers := []InferenceCheckpointLayer{cap: nn.info.layers.len}
	for i, layer in nn.info.layers {
		layer_type := nn.info.layer_types[i]
		variables := layer.variables()
		names := inference_checkpoint_parameter_names(layer_type, variables.len)!
		mut parameters := []InferenceCheckpointParameter{cap: variables.len}
		for j, variable in variables {
			mut values := []f64{len: variable.value.size()}
			for index in 0 .. variable.value.size() {
				values[index] = f64(variable.value.get_nth(index))
			}
			parameters << InferenceCheckpointParameter{
				name:   names[j]
				shape:  variable.value.shape.clone()
				dtype:  dtype
				values: values
			}
		}
		serialized_layers << InferenceCheckpointLayer{
			type_name:    layer_type
			config:       nn.info.layer_configs[i].clone()
			output_shape: layer.output_shape().clone()
			parameters:   parameters
		}
	}
	checkpoint := InferenceCheckpointFile{
		format:  inference_checkpoint_format
		version: inference_checkpoint_version
		model:   'Sequential'
		dtype:   dtype
		layers:  serialized_layers
	}
	inference_checkpoint_write_atomically(path, json2.encode(checkpoint))!
}

fn inference_checkpoint_write_atomically(path string, contents string) ! {
	temporary_path := os.join_path_single(os.dir(path),
		'.${os.file_name(path)}-${rand.uuid_v4()}')
	os.write_file(temporary_path, contents) or {
		write_error := err
		if os.exists(temporary_path) {
			os.rm(temporary_path) or {
				return error('inference checkpoint: write failed (${write_error.msg()}) and temporary-file cleanup failed (${err.msg()})')
			}
		}
		return write_error
	}
	inference_checkpoint_replace_file(temporary_path, path) or {
		replace_error := err
		os.rm(temporary_path) or {
			return error('inference checkpoint: replacement failed (${replace_error.msg()}) and temporary-file cleanup failed (${err.msg()})')
		}
		return replace_error
	}
}

fn inference_checkpoint_replace_file(source string, destination string) ! {
	if os.is_dir(destination) {
		return error('inference checkpoint: destination path is a directory')
	}
	$if windows {
		inference_checkpoint_replace_file_windows(source, destination)!
	} $else {
		os.rename(source, destination)!
	}
}

// load_inference_checkpoint validates a versioned CPU inference checkpoint
// against this already-constructed Sequential model before copying any weights.
pub fn (mut nn Sequential[T]) load_inference_checkpoint(path string) ! {
	data := os.read_file(path)!
	inference_checkpoint_validate_json(data)!
	checkpoint := json2.decode[InferenceCheckpointFile](data)!
	dtype := inference_checkpoint_validate_model[T](nn, false)!
	if checkpoint.format != inference_checkpoint_format
		|| checkpoint.version != inference_checkpoint_version {
		return error('inference checkpoint: unsupported format or version')
	}
	if checkpoint.model != 'Sequential' {
		return error('inference checkpoint: unsupported model type ${checkpoint.model}')
	}
	if checkpoint.dtype != dtype {
		return error('inference checkpoint: dtype mismatch: expected ${dtype}, got ${checkpoint.dtype}')
	}
	if checkpoint.layers.len != nn.info.layers.len {
		return error('inference checkpoint: layer count mismatch')
	}

	for i, layer in nn.info.layers {
		saved_layer := checkpoint.layers[i]
		layer_type := nn.info.layer_types[i]
		if saved_layer.type_name != layer_type {
			return error('inference checkpoint: layer type mismatch at index ${i}')
		}
		if saved_layer.output_shape != layer.output_shape() {
			return error('inference checkpoint: output shape mismatch at layer ${i}')
		}
		config := nn.info.layer_configs[i]
		if saved_layer.config.len != config.len {
			return error('inference checkpoint: layer configuration mismatch at index ${i}')
		}
		for key, value in config {
			if key !in saved_layer.config || saved_layer.config[key] != value {
				return error('inference checkpoint: layer configuration mismatch at index ${i}')
			}
		}
		variables := layer.variables()
		names := inference_checkpoint_parameter_names(layer_type, variables.len)!
		if saved_layer.parameters.len != variables.len {
			return error('inference checkpoint: parameter count mismatch at layer ${i}')
		}
		for j, variable in variables {
			parameter := saved_layer.parameters[j]
			if parameter.name != names[j] || parameter.dtype != dtype {
				return error('inference checkpoint: parameter identity or dtype mismatch at layer ${i}')
			}
			if parameter.shape != variable.value.shape
				|| parameter.values.len != variable.value.size() {
				return error('inference checkpoint: parameter shape mismatch at layer ${i}')
			}
			for value in parameter.values {
				converted := vtl.cast[T](value)
				converted_f64 := f64(converted)
				if math.is_nan(converted_f64) || math.is_inf(converted_f64, 0) {
					return error('inference checkpoint: parameter value is not finite for ${dtype}')
				}
			}
		}
	}

	for i, layer in nn.info.layers {
		for j, variable in layer.variables() {
			for index, value in checkpoint.layers[i].parameters[j].values {
				variable.value.set_nth(index, vtl.cast[T](value))
			}
		}
	}
}

fn inference_checkpoint_object(value json2.Any, field string) !map[string]json2.Any {
	match value {
		map[string]json2.Any {
			return value
		}
		else {
			return error('inference checkpoint: ${field} must be an object')
		}
	}
}

fn inference_checkpoint_array(value json2.Any, field string) ![]json2.Any {
	match value {
		[]json2.Any {
			return value
		}
		else {
			return error('inference checkpoint: ${field} must be an array')
		}
	}
}

fn inference_checkpoint_string(value json2.Any, field string) !string {
	match value {
		string {
			return value
		}
		else {
			return error('inference checkpoint: ${field} must be a string')
		}
	}
}

fn inference_checkpoint_integer(value json2.Any, field string) !int {
	match value {
		int {
			return value
		}
		f64 {
			return inference_checkpoint_float_integer(value, field)!
		}
		f32 {
			return inference_checkpoint_float_integer(f64(value), field)!
		}
		i64 {
			return int(value)
		}
		i32 {
			return int(value)
		}
		i16 {
			return int(value)
		}
		i8 {
			return int(value)
		}
		u64 {
			return int(value)
		}
		u32 {
			return int(value)
		}
		u16 {
			return int(value)
		}
		u8 {
			return int(value)
		}
		else {
			return error('inference checkpoint: ${field} must be an integer')
		}
	}
}

fn inference_checkpoint_float_integer(value f64, field string) !int {
	if math.is_nan(value) || math.is_inf(value, 0) || value != math.floor(value) {
		return error('inference checkpoint: ${field} must be a finite integer')
	}
	if sizeof(int) == 8 {
		if value < -9223372036854775808.0 || value >= 9223372036854775808.0 {
			return error('inference checkpoint: ${field} is outside the integer range')
		}
	} else {
		if value < -2147483648.0 || value > 2147483647.0 {
			return error('inference checkpoint: ${field} is outside the integer range')
		}
	}
	return int(value)
}

fn inference_checkpoint_number(value json2.Any, field string) !f64 {
	match value {
		f64 {
			return value
		}
		f32 {
			return f64(value)
		}
		int {
			return f64(value)
		}
		i64 {
			return f64(value)
		}
		i32 {
			return f64(value)
		}
		i16 {
			return f64(value)
		}
		i8 {
			return f64(value)
		}
		u64 {
			return f64(value)
		}
		u32 {
			return f64(value)
		}
		u16 {
			return f64(value)
		}
		u8 {
			return f64(value)
		}
		else {
			return error('inference checkpoint: ${field} must be numeric')
		}
	}
}

fn inference_checkpoint_exact_keys(object map[string]json2.Any, keys []string, field string) ! {
	if object.len != keys.len {
		return error('inference checkpoint: invalid fields in ${field}')
	}
	for key in keys {
		if key !in object {
			return error('inference checkpoint: missing ${field}.${key}')
		}
	}
	for key, _ in object {
		if key !in keys {
			return error('inference checkpoint: unknown ${field}.${key}')
		}
	}
}

fn inference_checkpoint_field(object map[string]json2.Any, key string, parent string) !json2.Any {
	return object[key] or {
		return error('inference checkpoint: missing ${parent}.${key}')
	}
}

fn inference_checkpoint_validate_json_shape(value json2.Any, field string) ! {
	shape := inference_checkpoint_array(value, field)!
	if shape.len == 0 {
		return error('inference checkpoint: ${field} must not be empty')
	}
	for index, dimension in shape {
		if inference_checkpoint_integer(dimension, '${field}[${index}]')! <= 0 {
			return error('inference checkpoint: ${field} dimensions must be positive')
		}
	}
}

fn inference_checkpoint_validate_json(data string) ! {
	root := inference_checkpoint_object(json2.decode[json2.Any](data)!, 'root')!
	inference_checkpoint_exact_keys(root, ['format', 'version', 'model', 'dtype', 'layers'], 'root')!
	if inference_checkpoint_string(inference_checkpoint_field(root, 'format', 'root')!,
		'format')! != inference_checkpoint_format {
		return error('inference checkpoint: unsupported format')
	}
	if inference_checkpoint_integer(inference_checkpoint_field(root, 'version', 'root')!,
		'version')! != inference_checkpoint_version {
		return error('inference checkpoint: unsupported version')
	}
	if inference_checkpoint_string(inference_checkpoint_field(root, 'model', 'root')!, 'model')! != 'Sequential' {
		return error('inference checkpoint: unsupported model type')
	}
	dtype := inference_checkpoint_string(inference_checkpoint_field(root, 'dtype', 'root')!, 'dtype')!
	if dtype !in ['float32', 'float64'] {
		return error('inference checkpoint: unsupported dtype ${dtype}')
	}
	layers := inference_checkpoint_array(inference_checkpoint_field(root, 'layers', 'root')!, 'layers')!
	if layers.len == 0 {
		return error('inference checkpoint: layers must not be empty')
	}
	for i, layer_value in layers {
		layer := inference_checkpoint_object(layer_value, 'layers[${i}]')!
		inference_checkpoint_exact_keys(layer, ['type_name', 'config', 'output_shape', 'parameters'],
			'layers[${i}]')!
		layer_type := inference_checkpoint_string(inference_checkpoint_field(layer, 'type_name',
			'layers[${i}]')!, 'layers[${i}].type_name')!
		if layer_type !in ['InputLayer', 'LinearLayer', 'ReLULayer'] {
			return error('inference checkpoint: unsupported layer type ${layer_type}')
		}
		config := inference_checkpoint_object(inference_checkpoint_field(layer, 'config', 'layers[${i}]')!,
			'layers[${i}].config')!
		for key, value in config {
			_ := inference_checkpoint_integer(value, 'layers[${i}].config.${key}')!
		}
		inference_checkpoint_validate_json_shape(inference_checkpoint_field(layer, 'output_shape',
			'layers[${i}]')!, 'layers[${i}].output_shape')!
		parameters := inference_checkpoint_array(inference_checkpoint_field(layer, 'parameters',
			'layers[${i}]')!, 'layers[${i}].parameters')!
		for j, parameter_value in parameters {
			parameter := inference_checkpoint_object(parameter_value, 'layers[${i}].parameters[${j}]')!
			inference_checkpoint_exact_keys(parameter, ['name', 'shape', 'dtype', 'values'],
				'layers[${i}].parameters[${j}]')!
			_ := inference_checkpoint_string(inference_checkpoint_field(parameter, 'name',
				'layers[${i}].parameters[${j}]')!, 'parameter name')!
			parameter_dtype := inference_checkpoint_string(inference_checkpoint_field(parameter, 'dtype',
				'layers[${i}].parameters[${j}]')!, 'parameter dtype')!
			if parameter_dtype != dtype {
				return error('inference checkpoint: inconsistent parameter dtype')
			}
			inference_checkpoint_validate_json_shape(inference_checkpoint_field(parameter, 'shape',
				'layers[${i}].parameters[${j}]')!, 'parameter shape')!
			values := inference_checkpoint_array(inference_checkpoint_field(parameter, 'values',
				'layers[${i}].parameters[${j}]')!, 'parameter values')!
			for k, value in values {
				number := inference_checkpoint_number(value, 'parameter value ${k}')!
				if math.is_nan(number) || math.is_inf(number, 0) {
					return error('inference checkpoint: parameter values must be finite')
				}
			}
		}
	}
}
