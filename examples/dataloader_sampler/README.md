# DataLoader subset sampler

Use a sampler with the tensor-backed `vtl.datasets.DataLoader` to train or
evaluate on a selected subset without copying the source tensor.
`SubsetRandomSampler` shuffles the requested row indices at each epoch.
`WeightedRandomSampler` draws rows in proportion to their weights, with
optional replacement. `FixedBatchSampler` wraps a sampler to set batch
boundaries and `drop_last`. The loader keeps feature rows and labels aligned,
and `collated_batch` exposes the default or custom-collated feature/label pair.

## Run

Run commands from `~/.vmodules`:

```sh
v -o /tmp/vtl_dataloader_sampler ./vtl/examples/dataloader_sampler/main.v
/tmp/vtl_dataloader_sampler
```

## Notes

The samplers are in `vtl.nn.data`. Use `SequentialSampler` for stable order or
`RandomSampler` for a shuffled full dataset. A custom sampler implements
`sample_indices(dataset_size) ![]int`; every returned index is checked before
the loader uses it. Weights need not sum to one; they must be finite,
non-negative, and include a positive value when samples are requested.
