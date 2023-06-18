module main

import vtl.datasets

imdb := datasets.load_imdb()!

assert imdb.train_features.shape == [25000]
assert imdb.test_features.shape == [25000]
assert imdb.train_labels.shape == [25000]
assert imdb.test_labels.shape == [25000]
