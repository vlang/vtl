# Complex least squares

Fit complex coefficients to an overdetermined system and inspect its residual,
numerical rank, and singular values. The example checks the recovered solution
against the known coefficients.

Run it from `~/.vmodules`:

```sh
systemd-run --user --scope --quiet -p MemoryMax=1G -p MemorySwapMax=0 --setenv=VJOBS=2 \
	v run ./vtl/examples/complex_least_squares/main.v
```

The current implementation converts the complex system into an equivalent
real-valued system. This keeps the solver dependency-free and uses more
temporary memory than the real least-squares path.
