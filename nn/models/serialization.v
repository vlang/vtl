module models

import json
import os
import vtl
import vtl.autograd
import vtl.nn.types

// Serialization format for a VTL model.
// Stores layer architecture and all variable tensors.
struct ModelState[T] {
	layer_types []string
	layer_shapes [][]int
	variable_tensors [][][]f64  // flattened data per variable
	variable_shapes [][]int
}

// save saves a Sequential model to a JSON file.
// Each variable's tensor data is stored as a flattened f64 array.
pub fn (nn &Sequential[T]) save(path string) ! {
	mut layer_types := []string{}
	mut layer_shapes := [][]int{}
	mut variable_tensors := [][][]f64{}
	mut variable_shapes := [][]int{}

	for layer in nn.info.layers {
		layer_types << typeof(layer).name
		layer_shapes << layer.output_shape()
		for v in layer.variables() {
			variable_shapes << v.value.shape
			mut data := []f64{}
			for i in 0 .. v.value.size() {
				data << f64(v.value.get_nth(i))
			}
			variable_tensors << [data]
		}
	}

	state := ModelState[T]{
		layer_types: layer_types
		layer_shapes: layer_shapes
		variable_tensors: variable_tensors
		variable_shapes: variable_shapes
	}
	state_json := json.encode(state)
	os.write_file(path, state_json)!
}

// load restores a Sequential model from a JSON file.
// Note: Reconstructs a new Sequential with the same architecture and loads weights.
// The model's layers must already be constructed in the same order as when saved.
pub fn (mut nn &Sequential[T]) load(path string) ! {
	data := os.read_file(path)!
	state := json.decode(ModelState[T], data)!

	mut var_idx := 0
	for i, layer in nn.info.layers {
		layer_name := state.layer_types[i]
		for j, v in layer.variables() {
			shape := state.variable_shapes[var_idx]
			flat_data := state.variable_tensors[var_idx][0]
			t := vtl.from_array(flat_data.map(vtl.cast[T](it)), shape)!
			// Copy data into the variable's value tensor
			for k, val in flat_data {
				v.value.set_nth(k, vtl.cast[T](val))
			}
			var_idx++
		}
	}
}

// save_variable saves a single autograd variable's tensor to a JSON file.
pub fn save_variable(path string, v &autograd.Variable[T]) ! {
	data := []f64{}
	for i in 0 .. v.value.size() {
		data << f64(v.value.get_nth(i))
	}
	state := {
		'shape': v.value.shape,
		'data': data
	}
	os.write_file(path, json.encode(state))!
}

// load_variable loads a tensor into an existing autograd variable from a JSON file.
pub fn load_variable(path string, mut v &autograd.Variable[T]) ! {
	data := os.read_file(path)!
	state := json.decode(map[string]interface{}, data)!
	shape := state['shape'] as []int
	flat_data := state['data'] as []f64
	t := vtl.from_array(flat_data.map(vtl.cast[T](it)), shape)!
	for i, val in flat_data {
		v.value.set_nth(i, vtl.cast[T](val))
	}
}