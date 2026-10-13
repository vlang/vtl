module models

import os
import vtl
import vtl.nn.optimizers
import vtl.nn.types

const nadam_training_checkpoint_test_dir = '/tmp/vtl_nadam_training_checkpoint_test'
const radam_training_checkpoint_test_dir = '/tmp/vtl_radam_training_checkpoint_test'

fn test_nadam_training_checkpoint_restores_schedule_and_moments() ! {
	os.mkdir_all(nadam_training_checkpoint_test_dir)!
	defer {
		os.rmdir_all(nadam_training_checkpoint_test_dir) or {}
	}
	mut model := sequential_with_layers[f64]([]types.Layer[f64]{})
	model.input([2])
	model.linear(1)
	mut optimizer := optimizers.nadam_optimizer[f64](
		learning_rate:          0.1
		weight_decay:           0.01
		momentum_decay:         0.01
		decoupled_weight_decay: true
	)
	optimizer.build_params(model.info.layers)
	for i, mut parameter in optimizer.params {
		parameter.grad = vtl.full[f64](parameter.value.shape, f64(i + 1))
	}
	optimizer.update()!
	path := '${nadam_training_checkpoint_test_dir}/nadam.json'
	model.save_nadam_training_checkpoint(path, 4, 0.2, optimizer)!

	mut restored := sequential_with_layers[f64]([]types.Layer[f64]{})
	restored.input([2])
	restored.linear(1)
	mut restored_optimizer := optimizers.nadam_optimizer[f64](learning_rate: 0.5)
	restored_optimizer.build_params(restored.info.layers)
	epoch, loss := restored.load_nadam_training_checkpoint(path, mut restored_optimizer)!
	assert epoch == 4
	assert loss == 0.2
	assert restored_optimizer.checkpoint_config() == optimizer.checkpoint_config()
	for i, parameter in optimizer.params {
		assert parameter.value.to_array() == restored_optimizer.params[i].value.to_array()
		assert optimizer.first_moments[i].to_array() == restored_optimizer.first_moments[i].to_array()
		assert optimizer.second_moments[i].to_array() == restored_optimizer.second_moments[i].to_array()
	}
}

fn test_radam_training_checkpoint_restores_step_and_moments() ! {
	os.mkdir_all(radam_training_checkpoint_test_dir)!
	defer {
		os.rmdir_all(radam_training_checkpoint_test_dir) or {}
	}
	mut model := sequential_with_layers[f64]([]types.Layer[f64]{})
	model.input([2])
	model.linear(1)
	mut optimizer := optimizers.radam_optimizer[f64](
		learning_rate:          0.1
		weight_decay:           0.01
		decoupled_weight_decay: true
	)
	optimizer.build_params(model.info.layers)
	for i, mut parameter in optimizer.params {
		parameter.grad = vtl.full[f64](parameter.value.shape, f64(i + 1))
	}
	optimizer.update()!
	path := '${radam_training_checkpoint_test_dir}/radam.json'
	model.save_radam_training_checkpoint(path, 4, 0.2, optimizer)!

	mut restored := sequential_with_layers[f64]([]types.Layer[f64]{})
	restored.input([2])
	restored.linear(1)
	mut restored_optimizer := optimizers.radam_optimizer[f64](learning_rate: 0.5)
	restored_optimizer.build_params(restored.info.layers)
	epoch, loss := restored.load_radam_training_checkpoint(path, mut restored_optimizer)!
	assert epoch == 4
	assert loss == 0.2
	assert restored_optimizer.checkpoint_config() == optimizer.checkpoint_config()
	for i, parameter in optimizer.params {
		assert parameter.value.to_array() == restored_optimizer.params[i].value.to_array()
		assert optimizer.first_moments[i].to_array() == restored_optimizer.first_moments[i].to_array()
		assert optimizer.second_moments[i].to_array() == restored_optimizer.second_moments[i].to_array()
	}
}
