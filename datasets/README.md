# VTL Datasets

VTL provides dataset loaders and batching utilities for ML examples and tests.

## Available datasets

| Dataset | Loader | Purpose |
|---------|--------|---------|
| MNIST | `datasets.load_mnist()` | Handwritten digit images (`28x28`) |
| IMDB | `datasets.load_imdb()` | Sentiment analysis reviews |
| CIFAR-10 | `datasets.load_cifar10(...)` | Image classification examples |
| DataLoader | `datasets.DataLoader[T]` | Batch, shuffle, and iterate tensors/labels |

DataLoader returns a view when a batch's sample indices form a contiguous range.
Shuffled, non-contiguous batches allocate tensors to preserve the requested
sample order. A non-positive `batch_size` produces an empty loader.

`DataLoaderConfig` defaults to `batch_size: 32`, `shuffle: true`, and
`drop_last: true`. These differ from PyTorch's defaults (`1`, `false`, and
`false`), so set all three explicitly when porting a training pipeline. Call
`reset()` between epochs to advance the deterministic shuffle order from the
configured seed.

For custom sample order, `new_data_loader_with_sampler_checked` and
`new_data_loader_with_labels_and_sampler_checked` accept samplers from
`vtl.nn.data`. Built-in options include sequential, random, subset-random,
weighted, and fixed batch sampling. Wrap a sampler with `FixedBatchSampler` to
choose batch boundaries and `drop_last` explicitly. Call `reset_checked()`
between epochs when using a sampler; it reports errors from custom sampler
implementations.

For custom batch construction, install a `CollateFn[T]` with
`DataLoader.set_collate_fn` and call `collated_batch(i)` or
`for_each_collated(...)`. The callback receives
the source feature tensor, an optional labels tensor, and a copy of the sample
indices selected for that batch. It returns a `CollatedBatch[T]` containing
the output features and optional labels. This supports dense tensor collation
such as transforms, padding, filtering, and reordering; it does not model
arbitrary Python objects or nested sample structures. See the
[custom collation example](../examples/datasets_custom_collate/main.v).

`load_imdb()` returns the full 25,000 review training and test splits. Use
`load_imdb_with_config` for a smaller, class-balanced subset. Review features
remain raw strings; labels are binary (`1` positive, `0` negative).

For simple NLP pipelines, `tokenize_text` lowercases and splits on Unicode
whitespace and ASCII punctuation while preserving UTF-8 text. `build_text_vocabulary`
creates deterministic frequency-ranked IDs; `<pad>` is ID `0` and `<unk>` is
ID `1`. `max_size` includes those reserved IDs, and `0` means no size limit.
This is a small preprocessing utility; it does not replace language-specific
tokenizers or normalization.

```v
import vtl.datasets

tokens := datasets.tokenize_text('A joyful, moving review!')
assert tokens == ['a', 'joyful', 'moving', 'review']

vocabulary := datasets.build_text_vocabulary([
	'joyful moving review',
	'moving review',
], 10000, 1)!
ids := vocabulary.encode('Joyful review')
assert vocabulary.decode(ids) == ['joyful', 'review']
```

```v
import vtl.datasets

imdb := datasets.load_imdb_with_config(datasets.ImdbConfig{
	train_count: 1000
	test_count:  200
})!
assert imdb.train_features.shape == [1000]
assert imdb.test_features.shape == [200]
```

## Examples

Run from `~/.vmodules`:

```bash
v run vtl/examples/datasets_mnist/main.v
v run vtl/examples/datasets_imdb/main.v
v run ./vtl/examples/nn_cifar10_tiny_synth/main.v
```

Use synthetic examples (`nn_cifar10_tiny_synth`,
`nn_cifar10_f32_tiny_synth`) for CI and quick local checks. Real dataset
examples may download/cache data and should be treated as local integration
tests.

## Related docs

- [Examples catalog](../examples/README.md)
- [Neural network tutorial](../docs/TUTORIAL_NEURAL_NETWORKS.md)
- [Lightweight development commands](../docs/DEV_LIGHTWEIGHT.md)
