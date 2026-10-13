module models

import os
import vtl
import vtl.nn.optimizers
import vtl.nn.types

const sgd_training_checkpoint_test_dir = '/tmp/vtl_sgd_training_checkpoint_test'

fn test_sgd_training_checkpoint_restores_momentum_and_model_state() ! {
	os.mkdir_all(sgd_training_checkpoint_test_dir)!
	defer {
		os.rmdir_all(sgd_training_checkpoint_test_dir) or {}
	}

	mut model := sequential_with_layers[f64]([]types.Layer[f64]{})
	model.input([2])
	model.linear(1)
	mut optimizer := optimizers.sgd[f64](
		learning_rate: 0.1
		momentum:      0.9
		dampening:     0.2
		weight_decay:  0.01
	)
	optimizer.build_params(model.info.layers)
	for i, mut parameter in optimizer.params {
		parameter.grad = vtl.full[f64](parameter.value.shape, f64(i + 1))
	}
	path := '${sgd_training_checkpoint_test_dir}/sgd_training_checkpoint.json'
	model.save_sgd_training_checkpoint(path, 5, 0.75, optimizer)!

	mut restored := sequential_with_layers[f64]([]types.Layer[f64]{})
	restored.input([2])
	restored.linear(1)
	mut restored_optimizer := optimizers.sgd[f64](learning_rate: 0.5)
	restored_optimizer.build_params(restored.info.layers)
	epoch, loss := restored.load_sgd_training_checkpoint(path, mut restored_optimizer)!
	assert epoch == 5
	assert loss == 0.75
	original_config := optimizer.checkpoint_config()
	restored_config := restored_optimizer.checkpoint_config()
	for key in original_config.keys() {
		assert restored_config[key] == original_config[key]
	}
	assert restored_optimizer.momentum_initialized == optimizer.momentum_initialized
	for i, parameter in optimizer.params {
		assert parameter.value.to_array() == restored_optimizer.params[i].value.to_array()
		assert optimizer.momentum_buffers[i].to_array() == restored_optimizer.momentum_buffers[i].to_array()
	}
	optimizer.update()!
	restored_optimizer.update()!
	assert restored_optimizer.momentum_initialized == optimizer.momentum_initialized
	for i, parameter in optimizer.params {
		assert parameter.value.to_array() == restored_optimizer.params[i].value.to_array()
		assert optimizer.momentum_buffers[i].to_array() == restored_optimizer.momentum_buffers[i].to_array()
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
		assert optimizer.momentum_buffers[i].to_array() == restored_optimizer.momentum_buffers[i].to_array()
	}
}
