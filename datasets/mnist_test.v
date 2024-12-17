// vtest retry: 3
import vtl.datasets

fn test_mnist() {
	unbuffer_stdout()
	println('start')
	mnist := datasets.load_mnist()!
	println('mnist dataset loaded')

	assert mnist.train_features.shape == [60000, 28, 28]
	assert mnist.test_features.shape == [10000, 28, 28]
	assert mnist.train_labels.shape == [60000]
	assert mnist.test_labels.shape == [10000]
	println('done')
}
