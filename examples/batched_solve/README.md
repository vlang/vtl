# Batched linear solve

Run from `~/.vmodules`:

```sh
systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 --setenv=VJOBS=2 \
	v run ./vtl/examples/batched_solve/main.v
```

This example solves two independent systems with one shared 1-D vector
right-hand side. It also broadcasts one unbatched matrix of right-hand sides
across both systems, following NumPy 2.0 `linalg.solve` shape rules.
