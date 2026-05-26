module main

import vtl
import vtl.autograd
import vtl.datasets
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

// Safe defaults (lower memory / shorter runtime)
const batch_size = 16
const epochs = 1
const learning_rate = 0.001
const train_count = 2000
const test_count = 500
const max_train_batches = 80

fn create_cnn_cifar10[T](ctx &autograd.Context[T]) &models.Sequential[T] {
	mut model := models.sequential_from_ctx[T](ctx)
	model.input([3, 32, 32])
	model.conv2d(3, 32, [3, 3], layers.Conv2DConfig{ padding: [1, 1] })
	model.relu()
	model.maxpool2d([2, 2], [0, 0], [2, 2])
	model.conv2d(32, 64, [3, 3], layers.Conv2DConfig{ padding: [1, 1] })
	model.relu()
	model.maxpool2d([2, 2], [0, 0], [2, 2])
	model.flatten()
	model.linear(128)
	model.relu()
	model.linear(10)
	model.softmax()
	return model
}

fn calculate_accuracy[T](predictions &vtl.Tensor[T], targets &vtl.Tensor[T]) f64 {
	shape := predictions.shape
	sample_count := shape[0]
	classes := shape[1]
	mut correct := 0
	for i := 0; i < sample_count; i++ {
		mut max_val := predictions.get([i, 0])
		mut max_idx := 0
		for j := 1; j < classes; j++ {
			v := predictions.get([i, j])
			if v > max_val {
				max_val = v
				max_idx = j
			}
		}
		mut target_idx := 0
		for j := 0; j < classes; j++ {
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

fn main() {
	ctx := autograd.ctx[f64]()

	println('Loading CIFAR-10 SAFE subset...')
	ds := datasets.load_cifar10_with_config(datasets.Cifar10Config{
		train_count: train_count
		test_count:  test_count
	})!

	println('Train samples: ${ds.train_features.shape[0]} | Test samples: ${ds.test_features.shape[0]}')

	mut model := create_cnn_cifar10[f64](ctx)
	model.cross_entropy_loss()

	mut optimizer := optimizers.adam_optimizer[f64](optimizers.AdamOptimizerConfig{
		learning_rate: learning_rate
	})
	optimizer.build_params(model.info.layers)

	// DataLoader for training with lockstep (features, labels)
	mut dl := datasets.new_data_loader_with_labels[f64](ds.train_features, ds.train_labels, datasets.DataLoaderConfig{
		batch_size: batch_size
		shuffle:    false
		drop_last:  true
		seed:       42
	})

	num_batches := dl.len()
	train_batches := if num_batches < max_train_batches { num_batches } else { max_train_batches }

	for epoch := 0; epoch < epochs; epoch++ {
		mut epoch_loss := 0.0
		mut epoch_correct := 0
		mut total_samples := 0

		for batch_id := 0; batch_id < train_batches; batch_id++ {
			feat, lab := dl.batch_with_labels(batch_id) or { break }

			x := ctx.variable(feat, requires_grad: true)
			y_pred := model.forward(x)!
			mut loss := model.loss(y_pred, lab)!
			loss_scalar := loss.value.get([0])
			epoch_loss += loss_scalar

			acc := calculate_accuracy(y_pred.value, lab)
			epoch_correct += int(f64(batch_size) * acc / 100.0)
			total_samples += batch_size

			loss.backprop()!
			optimizer.update()!

			if batch_id % 20 == 0 {
				println('Batch ${batch_id}/${train_batches} | Loss=${loss_scalar:.4f} | Acc=${acc:.2f}%')
			}
		}

		avg_loss := epoch_loss / f64(train_batches)
		accuracy := f64(epoch_correct) / f64(total_samples) * 100.0
		println('Epoch ${epoch + 1} done | Avg Loss=${avg_loss:.4f} | Avg Acc=${accuracy:.2f}%')

		dl.reset()
	}

	println('SAFE run finished.')
}
