module main

import vtl.datasets

mut mnist := datasets.load_mnist()!

assert mnist.train_features.shape == [60000, 28, 28]
assert mnist.test_features.shape == [10000, 28, 28]
assert mnist.train_labels.shape == [60000, 10]
assert mnist.test_labels.shape == [10000, 10]
