module main

import vtl
import vtl.autograd
import vtl.datasets
import vtl.nn.models
import vtl.nn.optimizers

// Ultra tiny settings (safe)
const batch_size = 8
const epochs = 1
const train_count = 64
const test_count = 16
const max_train_batches = 2

fn accuracy[T](pred &vtl.Tensor[T], target &vtl.Tensor[T]) f64 {
	rows := pred.shape[0]
	cols := pred.shape[1]
	mut ok := 0
	for i := 0; i < rows; i++ {
		mut pi := 0
		mut pv := pred.get([i, 0])
		for j := 1; j < cols; j++ {
			v := pred.get([i, j])
			if v > pv {
				pv = v
				pi = j
			}
		}
		mut ti := 0
		for j := 0; j < cols; j++ {
			if target.get([i, j]) > 0.5 {
				ti = j
				break
			}
		}
		if pi == ti {
			ok++
		}
	}
	return f64(ok) / f64(rows) * 100.0
}

fn main() {
	println('Loading CIFAR-10 tiny subset...')
	ds := datasets.load_cifar10_with_config(datasets.Cifar10Config{
		train_count: train_count
		test_count:  test_count
	})!

	ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([3, 32, 32])
	model.flatten()
	model.linear(10)
	model.softmax()
	model.cross_entropy_loss()

	mut opt := optimizers.adam_optimizer[f64](optimizers.AdamOptimizerConfig{
		learning_rate: 0.001
	})
	opt.build_params(model.info.layers)

	// Use DataLoader for features and labels in lockstep
	mut dl := datasets.new_data_loader_with_labels[f64](ds.train_features, ds.train_labels, datasets.DataLoaderConfig{
		batch_size: batch_size
		shuffle:    false // deterministic for tiny run
		drop_last:  true
		seed:       42
	})

	num_batches := dl.len()
	batches := if num_batches < max_train_batches { num_batches } else { max_train_batches }

	for epoch := 0; epoch < epochs; epoch++ {
		mut loss_sum := 0.0
		for b := 0; b < batches; b++ {
			feat, lab := dl.batch_with_labels(b) or { break }
			x := ctx.variable(feat, requires_grad: true)
			pred := model.forward(x)!
			mut loss := model.loss(pred, lab)!
			lv := loss.value.get([0])
			loss_sum += lv
			acc := accuracy(pred.value, lab)
			loss.backprop()!
			opt.update()!
			println('batch ${b + 1}/${batches} | loss=${lv:.4f} | acc=${acc:.2f}%')
		}
		dl.reset()
		println('epoch ${epoch + 1}/${epochs} | avg_loss=${loss_sum / f64(batches):.4f}')
	}

	println('Tiny CIFAR-10 run OK ✅')
}
