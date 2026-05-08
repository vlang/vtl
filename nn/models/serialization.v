module models

import json
import os
import vtl
import vtl.autograd

// ModelState stores all variable tensor data for serialization.
struct ModelState {
	layer_types      []string
	layer_shapes     [][]int
	variable_shapes  [][]int
	variable_tensors [][]f64 // each element is the flattened tensor data
}

// save saves a Sequential model's weights to a JSON file.
pub fn (nn &Sequential[T]) save(path string) ! {
	mut layer_types := []string{}
	mut layer_shapes := [][]int{}
	mut variable_shapes := [][]int{}
	mut variable_tensors := [][]f64{}

	for layer in nn.info.layers {
		layer_types << typeof(layer).name
		layer_shapes << layer.output_shape()
		for v in layer.variables() {
			variable_shapes << v.value.shape
			mut data := []f64{}
			for i in 0 .. v.value.size() {
				data << f64(v.value.get_nth(i))
			}
			variable_tensors << data
		}
	}

	state := ModelState{
		layer_types:      layer_types
		layer_shapes:     layer_shapes
		variable_shapes:  variable_shapes
		variable_tensors: variable_tensors
	}
	os.write_file(path, json.encode(state))!
}

// load restores weights into an existing Sequential model from a JSON file.
// The model's layers must already be constructed in the same order as when saved.
pub fn (nn &Sequential[T]) load(path string) ! {
	data := os.read_file(path)!
	state := json.decode(ModelState, data)!

	mut var_idx := 0
	for layer in nn.info.layers {
		for v in layer.variables() {
			if var_idx >= state.variable_shapes.len {
				break
			}
			shape := state.variable_shapes[var_idx]
			flat_data := state.variable_tensors[var_idx]
			t := vtl.from_array(flat_data.map(vtl.cast[T](it)), shape)!
			_ = t
			// Copy data into the variable's value tensor in-place
			mut v_val := unsafe { v }
			for k in 0 .. flat_data.len {
				v_val.value.set_nth(k, vtl.cast[T](flat_data[k]))
			}
			var_idx++
		}
	}
}
