module main

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models
import vtl.nn.optimizers
import os
import crypto.sha1
import net.http

// Training configuration
const batch_size = 64
const epochs = 10
const learning_rate = 0.001
const train_size = 50000
const test_size = 10000
const num_batches = train_size / batch_size
const channels = 3
const height = 32
const width = 32
const num_classes = 10
const cifar10_base = 'https://www.cs.toronto.edu/~kriz/'
const cifar10_file = 'cifar-10-binary.tar.gz'

// CIFAR-10 class names
const class_names = [
	'airplane',
	'automobile',
	'bird',
	'cat',
	'deer',
	'dog',
	'frog',
	'horse',
	'ship',
	'truck',
]

// Cifar10Dataset holds the CIFAR-10 dataset
struct Cifar10Dataset {
pub:
	train_features &vtl.Tensor[f64] = unsafe { nil }
	train_labels   &vtl.Tensor[f64] = unsafe { nil }
	test_features  &vtl.Tensor[f64] = unsafe { nil }
	test_labels    &vtl.Tensor[f64] = unsafe { nil }
}

fn get_cache_dir(subdir ...string) string {
	mut cache_dir := os.cache_dir()
	$if datasets_dir ? {
		cache_dir = datasets_dir
	}
	return os.join_path(cache_dir, ...subdir)
}

fn load_from_url(url string, target string) ! {
	datasets_cache_dir := get_cache_dir('datasets')
	if !os.is_dir(datasets_cache_dir) {
		os.mkdir_all(datasets_cache_dir)!
	}
	cache_file_name := sha1.hexhash(url)
	cache_file_path := if target == '' {
		os.join_path(datasets_cache_dir, cache_file_name)
	} else {
		target
	}
	if os.is_file(cache_file_path) {
		return
	}
	http.download_file(url, cache_file_path)!
}

fn download_dataset(dataset string, baseurl string, file string, uncompressed_dir string) !string {
	dataset_dir := os.real_path(get_cache_dir('datasets', dataset))
	target := os.join_path(dataset_dir, file)
	if !os.exists(target) {
		if !os.is_dir(dataset_dir) {
			os.mkdir_all(dataset_dir)!
		}
		load_from_url(url: '${baseurl}${file}', target: target)!
	}
	uncompressed_path := os.join_path(dataset_dir, uncompressed_dir)
	if !os.is_dir(uncompressed_path) {
		result := os.execute('tar -xvzf ${target} -C ${dataset_dir}')
		if result.exit_code != 0 {
			return error_with_code('Error extracting ${target}', result.exit_code)
		}
	}
	return uncompressed_path
}

// load_cifar10_batch loads a single batch file from CIFAR-10 binary format.
fn load_cifar10_batch(path string, expected_count int) !([]f64, []int) {
	if !os.exists(path) {
		return error('CIFAR-10 batch not found: ${path}')
	}
	data := os.read_file(path)!
	mut labels := []int{len: expected_count}
	mut images := []f64{len: expected_count * 3072}
	for i in 0 .. expected_count {
		offset := i * 3073
		labels[i] = int(data[offset])
		for j in 0 .. 3072 {
			images[i * 3072 + j] = f64(u8(data[offset + 1 + j])) / 255.0
		}
	}
	return images, labels
}

// load_cifar10_train_batches loads all 5 training batches
fn load_cifar10_train_batches(data_dir string) !([]f64, []int) {
	mut all_images := []f64{len: 0}
	mut all_labels := []int{len: 0}
	for batch_num := 1; batch_num <= 5; batch_num++ {
		batch_path := os.join_path(data_dir, 'cifar-10-batches-bin', 'data_batch_${batch_num}.bin')
		images, labels := load_cifar10_batch(batch_path, 10000)!
		all_images << images
		all_labels << labels
	}
	return all_images, all_labels
}

// load_cifar10 loads the CIFAR-10 dataset
fn load_cifar10() !Cifar10Dataset {
	dataset_path := download_dataset('cifar10', cifar10_base, cifar10_file, 'cifar-10-batches-bin')!
	println('Loading CIFAR-10 training data...')
	train_images, train_labels := load_cifar10_train_batches(dataset_path)!
	// Reshape training images to [N, C, H, W]
	train_features := vtl.from_1d(train_images)!.reshape([-1, channels, height, width])!
	// One-hot encode training labels
	mut train_labels_onehot := vtl.zeros[f64]([train_labels.len, num_classes])
	for i in 0 .. train_labels.len {
		class_idx := train_labels[i]
		if class_idx >= 0 && class_idx < num_classes {
			train_labels_onehot.set([i, class_idx], 1.0)
		}
	}
	train_labels_tensor := vtl.from_1d(train_labels_onehot.to_array()!)!.reshape([-1, num_classes])!
	// Load test data
	println('Loading CIFAR-10 test data...')
	test_path := os.join_path(dataset_path, 'cifar-10-batches-bin', 'test_batch.bin')
	test_images, test_labels := load_cifar10_batch(test_path, 10000)!
	// Reshape test images to [N, C, H, W]
	test_features := vtl.from_1d(test_images)!.reshape([-1, channels, height, width])!
	// One-hot encode test labels
	mut test_labels_onehot := vtl.zeros[f64]([test_labels.len, num_classes])
	for i in 0 .. test_labels.len {
		class_idx := test_labels[i]
		if class_idx >= 0 && class_idx < num_classes {
			test_labels_onehot.set([i, class_idx], 1.0)
		}
	}
	test_labels_tensor := vtl.from_1d(test_labels_onehot.to_array()!)!.reshape([-1, num_classes])!
	return Cifar10Dataset{
		train_features: train_features
		train_labels:   train_labels_tensor
		test_features:  test_features
		test_labels:    test_labels_tensor
	}
}

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
	for i in 0 .. batch_size {
		mut max_val := predictions.get([i, 0])
		mut max_idx := 0
		for j in 1 .. num_classes {
			val := predictions.get([i, j])
			if val > max_val {
				max_val = val
				max_idx = j
			}
		}
		mut target_idx := 0
		for j in 0 .. num_classes {
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
fn evaluate_validation[T](model &models.Sequential[T], ctx &autograd.Context[T], dataset &Cifar10Dataset) (f64, f64) {
	val_batch_size := 100
	val_num_batches := test_size / val_batch_size
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
	// Load CIFAR-10 dataset
	println('Loading CIFAR-10 dataset...')
	dataset := load_cifar10()!
	println('Dataset loaded successfully!')
	println('Training samples: ${dataset.train_features.shape[0]}')
	println('Test samples: ${dataset.test_features.shape[0]}')
	println('Image shape: ${dataset.train_features.shape[1..]}')
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
		println('  ${i + 1}   |   ${train_losses[i]:.4f}   |  ${train_accuracies[i]:.2f}%  |  ${val_losses[i]:.4f}  |  ${val_accuracies[i]:.2f}%')
	}
	println('----------------------------------------')
}
