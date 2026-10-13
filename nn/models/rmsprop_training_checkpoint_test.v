module models

import os
import vtl
import vtl.nn.optimizers
import vtl.nn.types

const rmsprop_training_checkpoint_test_dir = '/tmp/vtl_rmsprop_training_checkpoint_test'

fn test_rmsprop_training_checkpoint_restores_buffers_and_model_state() ! {
	os.mkdir_all(rmsprop_training_checkpoint_test_dir)!
	defer {
		os.rmdir_all(rmsprop_training_checkpoint_test_dir) or {}
	}

	mut model := sequential_with_layers[f64]([]types.Layer[f64]{})
	model.input([2])
	model.linear(1)
	mut optimizer := optimizers.rmsprop[f64](learning_rate: 0.1, alpha: 0.8, weight_decay: 0.01)
	optimizer.build_params(model.info.layers)
	for i, mut parameter in optimizer.params {
		parameter.grad = vtl.full[f64](parameter.value.shape, f64(i + 1))
	}
	optimizer.update()!
	path := '${rmsprop_training_checkpoint_test_dir}/rmsprop_training_checkpoint.json'
	model.save_rmsprop_training_checkpoint(path, 3, 0.25, optimizer)!

	mut restored := sequential_with_layers[f64]([]types.Layer[f64]{})
	restored.input([2])
	restored.linear(1)
	mut restored_optimizer := optimizers.rmsprop[f64](learning_rate: 0.5)
	restored_optimizer.build_params(restored.info.layers)
	epoch, loss := restored.load_rmsprop_training_checkpoint(path, mut restored_optimizer)!
	assert epoch == 3
	assert loss == 0.25
	assert restored_optimizer.checkpoint_config() == optimizer.checkpoint_config()
	for i, parameter in optimizer.params {
		assert parameter.value.to_array() == restored_optimizer.params[i].value.to_array()
		assert optimizer.sq_avg[i].to_array() == restored_optimizer.sq_avg[i].to_array()
	}

	for i, mut parameter in optimizer.params {
		parameter.grad = vtl.full[f64](parameter.value.shape, f64(i + 2))
		mut restored_parameter := restored_optimizer.params[i]
		restored_parameter.grad = vtl.full[f64](parameter.value.shape, f64(i + 2))
	}
	optimizer.update()!
	restored_optimizer.update()!
	for i, parameter in optimizer.params {
		assert parameter.value.to_array() == restored_optimizer.params[i].value.to_array()
		assert optimizer.sq_avg[i].to_array() == restored_optimizer.sq_avg[i].to_array()
	}
}
