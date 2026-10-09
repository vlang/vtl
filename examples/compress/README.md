# Compress values and tensor axes

Filter a tensor in row-major order with `compress`, or select rows and columns with
`compress_axis` while preserving rank.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/compress/main.v
```

## Notes

The condition length must match the flattened input for global compression or the selected axis
length for axis compression.
