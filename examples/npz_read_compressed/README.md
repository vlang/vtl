# Read a named array from NPZ

Read and list members of a compressed or uncompressed NumPy `.npz` archive, then load its `weights` member as `f64`.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/npz_read_compressed/main.v path/to/archive.npz
```

## Notes

The supplied archive must contain a numeric `weights` array compatible with `f64`. The reader does not extract archive paths.
