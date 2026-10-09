# Set membership for tensor values

Build an elementwise membership mask with `isin` and use the mask to select accepted measurements.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/set_membership/main.v
```

## Notes

The output retains input order and duplicate occurrences from the measurements tensor.
