# Batched linear solve

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/batched_solve/main.v
```

This example solves two independent systems. It also broadcasts one matrix of
right-hand sides across both systems by giving it a leading size-one batch
dimension, following NumPy's `linalg.solve` shape rules.
