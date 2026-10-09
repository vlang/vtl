# Unique values and metadata

Compute sorted unique values, occurrence counts, inverse indices, first positions, and unique rows
along an axis.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/unique/main.v
```

## Notes

Use the returned inverse indices to reconstruct the original values from the unique array.
Row-wise uniqueness treats each row as one item.
