module main

import vtl
import vtl.autograd
import vtl.datasets
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers

// Training configuration
const batch_size = 64
const epochs = 10
const learning_rate = 0.001

// create_cnn_cifar10 creates a CNN for CIFAR-10 classification
fn create_cnn_cifar10[T](ctx &autograd.Context[T]) &models.Sequential[T] {
	mut model := models.sequential_from_ctx[T](ctx)
	// Input shape: [batch, 3, 32, 32]
	model.input([3, 32, 32])
	// Block 1: Conv2d (3, 64, 3x3) + ReLU + MaxPool
	model.conv2d(3, 64, [3, 3], layers.Conv2DConfig{ padding: [1, 1] })
	model.relu()
	model.maxpool2d([2, 2], [0, 0], [2, 2])
	// Block 2: Conv2d (64, 128, 3x3) + ReLU + MaxPool
	model.conv2d(64, 128, [3, 3], layers.Conv2DConfig{ padding: [1, 1] })
	model.relu()
	model.maxpool2d([2, 2], [0, 0], [2, 2])
	// Block 3: Conv2d (128, 256, 3x3) + ReLU + MaxPool
	model.conv2d(128, 256, [3, 3], layers.Conv2DConfig{ padding: [1, 1] })
	model.relu()
	model.maxpool2d([2, 2], [0, 0], [2, 2])
	// Flatten: [batch, 256*4*4] = [batch, 4096]
	model.flatten()
	// FC1: Linear (4096, 256) + ReLU
	model.linear(256)
	model.relu()
	// FC2: Linear (256, 10) + Softmax
	model.linear(10)
	model.softmax()
	return model
}

// calculate_accuracy calculates classification accuracy
fn calculate_accuracy[T](predictions &vtl.Tensor[T], targets &vtl.Tensor[T]) f64 {
	pred_shape := predictions.shape
	batch_size := pred_shape[0]
	num_classes := pred_shape[1]
	mut correct := 0
	for i := 0; i < batch_size; i++ {
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
	return f64(correct) / f64(batch_size) * 100.0
}

// evaluate_validation runs validation and returns loss and accuracy
fn evaluate_validation[T](model &models.Sequential[T], ctx &autograd.Context[T], dataset &datasets.Cifar10Dataset) (f64, f64) {
	val_batch_size := 100
	val_num_batches := 10000 / val_batch_size
	mut total_loss := 0.0
	mut total_correct := 0
	mut total_samples := 0
	for batch_id := 0; batch_id < int(val_num_batches); batch_id++ {
		offset := batch_id * val_batch_size
		x := ctx.variable(dataset.test_features.slice([offset, offset + val_batch_size])!,
			requires_grad: false
		)
		y := dataset.test_labels.slice([offset, offset + val_batch_size])!
		y_pred := model.forward(x)!
		loss := model.loss(y_pred, y)!
		acc := calculate_accuracy(y_pred.value, y)
		total_loss += loss.value
		total_correct += int(f64(val_batch_size) * acc / 100.0)
		total_samples += val_batch_size
	}
	avg_loss := total_loss / f64(val_num_batches)
	accuracy := f64(total_correct) / f64(total_samples) * 100.0
	return avg_loss, accuracy
}

fn main() {
	ctx := autograd.ctx[f64]()

	// Load CIFAR-10 dataset using the datasets module
	println('Loading CIFAR-10 dataset...')
	dataset := datasets.load_cifar10()!
	println('Dataset loaded successfully!')
	println('Training samples: ${dataset.train_features.shape[0]}')
	println('Test samples: ${dataset.test_features.shape[0]}')
	println('Image shape: ${dataset.train_features.shape[1..]}')
	println('Class names: ${datasets.class_names()}')

	// Create model
	println('\nBuilding CNN model...')
	mut model := create_cnn_cifar10[f64](ctx)
	// Set loss function
	model.cross_entropy_loss()

	// Create optimizer
	mut optimizer := optimizers.adam_optimizer[f64](optimizers.AdamOptimizerConfig{
		learning_rate: learning_rate
	})
	optimizer.build_params(model.info.layers)
	println('Model built successfully!')

	num_batches := dataset.train_features.shape[0] / batch_size
	println('Training for ${epochs} epochs with batch size ${batch_size}')

	// Training history
	mut train_losses := []f64{}
	mut train_accuracies := []f64{}
	mut val_losses := []f64{}
	mut val_accuracies := []f64{}

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		println('\n========================================')
		println('Epoch ${epoch + 1}/${epochs}')
		println('========================================')
		mut epoch_loss := 0.0
		mut epoch_correct := 0
		mut total_samples := 0

		// Mini-batch training
		for batch_id := 0; batch_id < int(num_batches); batch_id++ {
			offset := batch_id * batch_size
			x := ctx.variable(dataset.train_features.slice([offset, offset + batch_size])!,
				requires_grad: true
			)
			y := dataset.train_labels.slice([offset, offset + batch_size])!
			y_pred := model.forward(x)!
			mut loss := model.loss(y_pred, y)!
			epoch_loss += loss.value
			acc := calculate_accuracy(y_pred.value, y)
			epoch_correct += int(f64(batch_size) * acc / 100.0)
			total_samples += batch_size

			if batch_id % 100 == 0 {
				println('  Batch ${batch_id}/${num_batches}: Loss=${loss.value:.4f}, Acc=${acc:.2f}%')
			}

			loss.backprop()!
			optimizer.update()!
		}

		avg_loss := epoch_loss / f64(num_batches)
		accuracy := f64(epoch_correct) / f64(total_samples) * 100.0
		println('\n  Epoch ${epoch + 1} Summary:')
		println('    Train Loss: ${avg_loss:.4f}')
		println('    Train Accuracy: ${accuracy:.2f}%')
		train_losses << avg_loss
		train_accuracies << accuracy

		// Validation
		println('\n  Running validation...')
		val_loss, val_acc := evaluate_validation(model, ctx, dataset)
		val_losses << val_loss
		val_accuracies << val_acc
		println('    Val Loss: ${val_loss:.4f}')
		println('    Val Accuracy: ${val_acc:.2f}%')
	}

	// Print final summary
	println('\n========================================')
	println('Training Complete!')
	println('========================================')
	println('\nTraining History:')
	println('----------------------------------------')
	println('Epoch | Train Loss | Train Acc | Val Loss | Val Acc')
	println('----------------------------------------')
	for i := 0; i < epochs; i++ {
		println('  ${i + 1:3d} |    ${train_losses[i]:.4f}   |  ${train_accuracies[i]:.2f}%  |  ${val_losses[i]:.4f}  |  ${val_accuracies[i]:.2f}%')
	}
	println('----------------------------------------')
}
