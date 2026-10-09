# Weighted averages

Compute a global weighted mean using elementwise weights and row-wise weighted means using shared
per-column reliability weights.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/weighted_average/main.v
```

## Notes

Weights must be non-negative and have a nonzero sum for each reduced slice. See [reductions
tutorial](../../docs/TUTORIAL_REDUCTIONS.md).
