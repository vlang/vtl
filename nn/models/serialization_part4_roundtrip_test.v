module models

import os
import math
import vtl
import vtl.nn.types

const test_model_dir_4d = '/tmp/vtl_test_models_4d'

fn setup_test_dir_4d() string {
	os.mkdir_all(test_model_dir_4d) or { panic(err) }
	return test_model_dir_4d
}

fn cleanup_test_dir_4d() {
	os.rmdir_all(test_model_dir_4d) or {}
}

fn test_round_trip_weights_allclose_1e9() {
	test_dir := setup_test_dir_4d()
	defer {
		cleanup_test_dir_4d()
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
