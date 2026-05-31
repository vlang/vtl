module main

import os
import vtl
import vtl.autograd
import vtl.datasets
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

// Training configuration
const batch_size = 64
const epochs = 1
const learning_rate = 0.001
const checkpoint_dir = 'checkpoints'
const checkpoint_every_epochs = 1

fn create_cnn_cifar10[T](ctx &autograd.Context[T]) &models.Sequential[T] {
	mut model := models.sequential_from_ctx[T](ctx)
	model.input([3, 32, 32])
	model.conv2d(3, 64, [3, 3], layers.Conv2DConfig{ padding: [1, 1] })
	model.relu()
	model.maxpool2d([2, 2], [0, 0], [2, 2])
	model.conv2d(64, 128, [3, 3], layers.Conv2DConfig{ padding: [1, 1] })
	model.relu()
	model.maxpool2d([2, 2], [0, 0], [2, 2])
	model.conv2d(128, 256, [3, 3], layers.Conv2DConfig{ padding: [1, 1] })
	model.relu()
	model.maxpool2d([2, 2], [0, 0], [2, 2])
	model.flatten()
	model.linear(256)
	model.relu()
	model.linear(10)
	model.softmax()
	return model
}

fn calculate_accuracy[T](predictions &vtl.Tensor[T], targets &vtl.Tensor[T]) f64 {
	pred_shape := predictions.shape
	sample_count := pred_shape[0]
	num_classes := pred_shape[1]
	mut correct := 0
	for i := 0; i < sample_count; i++ {
		mut max_val := predictions.get([i, 0])
		mut max_idx := 0
		for j := 1; j < num_classes; j++ {
			val := predictions.get([i, j])
			if val > max_val {
				max_val = val
				max_idx = j
			}
		}
		mut target_idx := 0
		for j := 0; j < num_classes; j++ {
			if targets.get([i, j]) > 0.5 {
				target_idx = j
				break
			}
		}
		if max_idx == target_idx {
			correct++
		}
	}
	return f64(correct) / f64(sample_count) * 100.0
}

fn wants_resume() bool {
	for arg in os.args {
		if arg == '--resume' {
			return true
		}
	}
	return false
}

fn checkpoint_path(epoch int) string {
	return '${checkpoint_dir}/cifar10_epoch_${epoch}.json'
}

fn latest_checkpoint() !string {
	if !os.exists(checkpoint_dir) {
		return error('no checkpoint directory')
	}
	mut best_epoch := -1
	mut best_path := ''
	entries := os.ls(checkpoint_dir)!
	for entry in entries {
		if !entry.ends_with('.json') {
			continue
		}
		if entry.starts_with('cifar10_epoch_') {
			suffix := entry.all_after('cifar10_epoch_').all_before('.json')
			epoch := suffix.int()
			if epoch > best_epoch {
				best_epoch = epoch
				best_path = os.join_path(checkpoint_dir, entry)
			}
		}
	}
	if best_path == '' {
		return error('no checkpoint files found')
	}
	return best_path
}

fn main() {
	ctx := autograd.ctx[f64]()
	os.mkdir_all(checkpoint_dir) or {}

	println('Loading CIFAR-10 dataset...')
	ds := datasets.load_cifar10_with_config(datasets.Cifar10Config{
		train_count: 10000
		test_count:  2000
	})!
	println('Dataset loaded successfully!')
	println('Training samples: ${ds.train_features.shape[0]}')
	println('Test samples: ${ds.test_features.shape[0]}')
	println('Image shape: ${ds.train_features.shape[1..]}')
	println('Class names: ${datasets.class_names()}')

	println('\nBuilding CNN model...')
	mut model := create_cnn_cifar10[f64](ctx)
	model.cross_entropy_loss()

	mut optimizer := optimizers.adam_optimizer[f64](optimizers.AdamOptimizerConfig{
		learning_rate: learning_rate
	})
	optimizer.build_params(model.info.layers)
	println('Model built successfully!')

	mut start_epoch := 0
	if wants_resume() {
		path := latest_checkpoint()!
		model.load_weights(path)!
		epoch_meta, loss_meta := models.Sequential.load_checkpoint[f64](path)!
		start_epoch = epoch_meta
		println('Resumed from ${path} (epoch ${epoch_meta}, loss ${loss_meta:.6f})')
	}

	// Training DataLoader with lockstep features + labels
	mut train_dl := datasets.new_data_loader_with_labels[f64](ds.train_features, ds.train_labels, datasets.DataLoaderConfig{
		batch_size: batch_size
		shuffle:    false
		drop_last:  true
		seed:       42
	})

	// Validation DataLoader (drop_last=false to use all samples)
	val_dl := datasets.new_data_loader_with_labels[f64](ds.test_features, ds.test_labels, datasets.DataLoaderConfig{
		batch_size: 100
		shuffle:    false
		drop_last:  false
		seed:       0
	})

	println('Training for ${epochs} epoch(s) with batch size ${batch_size}')
	println('Train batches: ${train_dl.len()} | Val batches: ${val_dl.len()}')

	mut train_losses := []f64{}
	mut train_accuracies := []f64{}
	mut val_losses := []f64{}
	mut val_accuracies := []f64{}

	for epoch := start_epoch; epoch < epochs; epoch++ {
		println('\n========================================')
		println('Epoch ${epoch + 1}/${epochs}')
		println('========================================')

		mut epoch_loss := 0.0
		mut epoch_correct := 0
		mut total_samples := 0

		for batch_id := 0; batch_id < train_dl.len(); batch_id++ {
			feat, lab := train_dl.batch_with_labels(batch_id) or { break }

			x := ctx.variable(feat, requires_grad: true)
			y_pred := model.forward(x)!
			mut loss := model.loss(y_pred, lab)!
			loss_scalar := loss.value.get([0])
			epoch_loss += loss_scalar

			acc := calculate_accuracy(y_pred.value, lab)
			epoch_correct += int(f64(batch_size) * acc / 100.0)
			total_samples += batch_size

			if batch_id % 50 == 0 {
				println('  Batch ${batch_id}/${train_dl.len()}: Loss=${loss_scalar:.4f}, Acc=${acc:.2f}%')
			}

			loss.backprop()!
			optimizer.update()!
		}

		avg_loss := epoch_loss / f64(train_dl.len())
		accuracy := f64(epoch_correct) / f64(total_samples) * 100.0
		println('\n  Epoch ${epoch + 1} Summary:')
		println('    Train Loss: ${avg_loss:.4f}')
		println('    Train Accuracy: ${accuracy:.2f}%')
		train_losses << avg_loss
		train_accuracies << accuracy

		// Validation using DataLoader
		println('\n  Running validation...')
		mut val_loss_sum := 0.0
		mut val_correct := 0
		mut val_total := 0
		for val_batch_id := 0; val_batch_id < val_dl.len(); val_batch_id++ {
			vfeat, vlab := val_dl.batch_with_labels(val_batch_id) or { break }
			vx := ctx.variable(vfeat, requires_grad: false)
			vpred := model.forward(vx)!
			vloss := model.loss(vpred, vlab)!
			vacc := calculate_accuracy(vpred.value, vlab)
			batch_sz := vfeat.shape[0]
			val_loss_sum += vloss.value.get([0])
			val_correct += int(f64(batch_sz) * vacc / 100.0)
			val_total += batch_sz
		}
		val_avg_loss := val_loss_sum / f64(val_dl.len())
		val_accuracy := f64(val_correct) / f64(val_total) * 100.0
		println('    Val Loss: ${val_avg_loss:.4f}')
		println('    Val Accuracy: ${val_accuracy:.2f}%')
		val_losses << val_avg_loss
		val_accuracies << val_accuracy

		train_dl.reset()

		if (epoch + 1) % checkpoint_every_epochs == 0 {
			ckpt := checkpoint_path(epoch + 1)
			model.save_checkpoint(ckpt, epoch + 1, avg_loss)!
			println('    Checkpoint saved: ${ckpt}')
		}
	}

	println('\n========================================')
	println('Training Complete!')
	println('========================================')
	if train_losses.len > 0 {
		println('\nTraining History:')
		println('----------------------------------------')
		println('Epoch | Train Loss | Train Acc | Val Loss | Val Acc')
		println('----------------------------------------')
		for i := 0; i < train_losses.len; i++ {
			println('  ${i + 1:3d} |    ${train_losses[i]:.4f}   |  ${train_accuracies[i]:.2f}%  |  ${val_losses[i]:.4f}  |  ${val_accuracies[i]:.2f}%')
		}
		println('----------------------------------------')
	}
}
