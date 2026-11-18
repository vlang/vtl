module models

import json2
import os
import vtl
import vtl.autograd

fn inference_checkpoint_test_dir() !string {
	dir := os.join_path(os.getwd(), '.vtl_inference_checkpoint_test_${os.getpid()}')
	os.mkdir_all(dir)!
	return dir
}

fn inference_checkpoint_test_model(ctx &autograd.Context[f64]) &Sequential[f64] {
	mut model := sequential_from_ctx[f64](ctx)
	model.input([2])
	model.linear(1)
	model.relu()
	variables := model.info.layers[1].variables()
	variables[0].value.set_nth(0, 0.25)
	variables[0].value.set_nth(1, -0.5)
	variables[1].value.set_nth(0, 0.1)
	return model
}

fn inference_checkpoint_test_model_f32(ctx &autograd.Context[f32]) &Sequential[f32] {
	mut model := sequential_from_ctx[f32](ctx)
	model.input([2])
	model.linear(1)
	model.relu()
	variables := model.info.layers[1].variables()
	variables[0].value.set_nth(0, f32(0.25))
	variables[0].value.set_nth(1, f32(-0.5))
	variables[1].value.set_nth(0, f32(0.1))
	return model
}

fn test_inference_checkpoint_integer_accepts_whole_float_tokens_and_rejects_unsafe_values() ! {
	assert inference_checkpoint_integer(json2.decode[json2.Any]('4.0')!, 'test')! == 4
	for token in ['1.5', '1e300', '-1e300'] {
		value := json2.decode[json2.Any](token)!
		mut rejected := false
		inference_checkpoint_integer(value, 'test') or {
			rejected = true
		}
		assert rejected
	}
}

fn test_inference_checkpoint_f32_round_trip() ! {
	dir := inference_checkpoint_test_dir()!
	defer {
		os.rmdir_all(dir) or {}
	}
	path := os.join_path(dir, 'model.json')
	source_ctx := autograd.ctx[f32]()
	source := inference_checkpoint_test_model_f32(source_ctx)
	source_input := source_ctx.variable(vtl.from_array[f32]([2.0, 1.0], [1, 2])!,
		requires_grad: false
	)
	expected := source.forward(source_input)!.value.to_array()
	source.save_inference_checkpoint(path)!

	restored_ctx := autograd.ctx[f32]()
	mut restored := inference_checkpoint_test_model_f32(restored_ctx)
	restored.load_inference_checkpoint(path)!
	actual_input := restored_ctx.variable(vtl.from_array[f32]([2.0, 1.0], [1, 2])!,
		requires_grad: false
	)
	actual := restored.forward(actual_input)!.value.to_array()
	assert actual == expected
}

fn test_inference_checkpoint_round_trip() ! {
	dir := inference_checkpoint_test_dir()!
	defer {
		os.rmdir_all(dir) or {}
	}
	path := os.join_path(dir, 'model.json')
	ctx := autograd.ctx[f64]()
	model := inference_checkpoint_test_model(ctx)
	input := ctx.variable(vtl.from_array[f64]([2.0, 1.0], [1, 2])!, requires_grad: false)
	expected := model.forward(input)!.value.to_array()
	model.save_inference_checkpoint(path)!

	mut restored_ctx := autograd.ctx[f64]()
	mut restored := inference_checkpoint_test_model(restored_ctx)
	restored.info.layers[1].variables()[0].value.set_nth(0, 99.0)
	restored.load_inference_checkpoint(path)!
	restored.eval()
	restored_ctx.no_grad = true
	actual_input := restored_ctx.variable(vtl.from_array[f64]([2.0, 1.0], [1, 2])!,
		requires_grad: false
	)
	actual := restored.forward(actual_input)!.value.to_array()
	assert actual == expected
}

fn test_inference_checkpoint_save_replaces_existing_file() ! {
	dir := inference_checkpoint_test_dir()!
	defer {
		os.rmdir_all(dir) or {}
	}
	path := os.join_path(dir, 'model.json')
	mut model := inference_checkpoint_test_model(autograd.ctx[f64]())
	model.save_inference_checkpoint(path)!
	original := os.read_file(path)!

	model.info.layers[1].variables()[0].value.set_nth(0, 0.75)
	model.save_inference_checkpoint(path)!

	assert os.read_file(path)! != original
	entries := os.ls(dir)!
	assert entries == ['model.json']
}

fn test_inference_checkpoint_save_cleans_temporary_file_on_replace_failure() ! {
	dir := inference_checkpoint_test_dir()!
	defer {
		os.rmdir_all(dir) or {}
	}
	path := os.join_path(dir, 'occupied')
	os.mkdir(path)!
	marker := os.join_path(path, 'keep')
	os.write_file(marker, 'unchanged')!
	model := inference_checkpoint_test_model(autograd.ctx[f64]())
	if _ := model.save_inference_checkpoint(path) {
		assert false, 'saving over a directory must fail'
	}
	assert os.read_file(marker)! == 'unchanged'
	assert os.ls(path)! == ['keep']
	entries := os.ls(dir)!
	assert entries == ['occupied']
}

fn test_inference_checkpoint_rejects_bad_schema_version_and_dtype() ! {
	dir := inference_checkpoint_test_dir()!
	defer {
		os.rmdir_all(dir) or {}
	}
	path := os.join_path(dir, 'model.json')
	model := inference_checkpoint_test_model(autograd.ctx[f64]())
	model.save_inference_checkpoint(path)!
	original := os.read_file(path)!

	for invalid in [
		original.replace('"version":1', '"version":2'),
		original.replace('"dtype":"float64"', '"dtype":"float32"'),
		original.replace('"model":"Sequential"', '"model":"Other"'),
		original.replace('"type_name":"LinearLayer"', '"type_name":"ReLULayer"'),
		original.replace('"version":1', '"version":1,"unexpected":true'),
	] {
		os.write_file(path, invalid)!
		mut restored := inference_checkpoint_test_model(autograd.ctx[f64]())
		mut rejected := false
		restored.load_inference_checkpoint(path) or {
			rejected = true
		}
		assert rejected
	}
}

fn test_inference_checkpoint_rejects_incompatible_layers_and_non_finite_weights() ! {
	dir := inference_checkpoint_test_dir()!
	defer {
		os.rmdir_all(dir) or {}
	}
	path := os.join_path(dir, 'model.json')
	model := inference_checkpoint_test_model(autograd.ctx[f64]())
	model.save_inference_checkpoint(path)!
	original := json2.decode[InferenceCheckpointFile](os.read_file(path)!)!

	original_data := json2.encode(original)
	wrong_shape := original_data.replace('"output_shape":[2]', '"output_shape":[3]')
	assert wrong_shape != original_data
	os.write_file(path, wrong_shape)!
	mut shape_model := inference_checkpoint_test_model(autograd.ctx[f64]())
	mut shape_rejected := false
	shape_model.load_inference_checkpoint(path) or {
		shape_rejected = true
	}
	assert shape_rejected

	parameter_shape := original_data.replace('"shape":[1,2]', '"shape":[1,3]')
	assert parameter_shape != original_data
	os.write_file(path, parameter_shape)!
	mut compatible_model := inference_checkpoint_test_model(autograd.ctx[f64]())
	mut parameter_shape_rejected := false
	compatible_model.load_inference_checkpoint(path) or {
		parameter_shape_rejected = true
	}
	assert parameter_shape_rejected

	non_finite_data := json2.encode(original).replace('0.25', '1e309')
	assert non_finite_data != json2.encode(original)
	os.write_file(path, non_finite_data)!
	mut restored := inference_checkpoint_test_model(autograd.ctx[f64]())
	mut value_rejected := false
	restored.load_inference_checkpoint(path) or {
		value_rejected = true
	}
	assert value_rejected
}
