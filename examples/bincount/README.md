# Count integer labels

Count non-negative integer labels with `stats.bincount`, then apply per-observation weights with
`stats.bincount_weighted`.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/bincount/main.v
```

## Notes

The requested output length preserves trailing empty bins. The weighted form returns
floating-point totals.
