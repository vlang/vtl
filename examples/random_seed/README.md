# Reproducible random samples

Reset the global VTL random seed and verify that uniform and several distribution samplers repeat
their sequences.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/random_seed/main.v
```

## Notes

Reproducibility applies to the same runtime and algorithm version; it is not a cross-version
bitstream guarantee.
