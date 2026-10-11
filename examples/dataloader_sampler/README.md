# DataLoader subset sampler

Use a sampler to train or evaluate on a selected subset without copying the
source matrix. `SubsetRandomSampler` shuffles the requested row indices at each
epoch, and `DataLoader` keeps feature rows and labels aligned.

## Run

Run commands from `~/.vmodules`:

```sh
v -o /tmp/vtl_dataloader_sampler ./vtl/examples/dataloader_sampler/main.v
/tmp/vtl_dataloader_sampler
```

## Notes

Use `SequentialSampler` for stable order or `RandomSampler` for a shuffled full
dataset. A custom sampler implements `sample_indices(dataset_size) ![]int`;
every returned index is checked before the loader uses it.
