# Complex linear system

This example solves `A * x = b` with complex128 tensors. The coefficient
matrix has a zero in its first diagonal position, so the solve exercises
partial pivoting.

Run it from `~/.vmodules`:

```sh
systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 --setenv=VJOBS=2 \
	v run ./vtl/examples/complex_linear_solve/main.v
```

The `vtl.la.solve_complex` API also accepts matrix right-hand sides and
broadcastable leading batch dimensions. It returns complex128 values and an
error for singular systems.
