# Statistics over multiple axes

Reduce a 3-D tensor over multiple axes for mean and variance, including `keepdims`, then ignore
NaNs with `nanmean_along_axes`.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/multi_axis_stats/main.v
```

## Notes

Negative axes are accepted. Reduced dimensions are retained only when `keepdims` is true; see
[reductions tutorial](../../docs/TUTORIAL_REDUCTIONS.md).
