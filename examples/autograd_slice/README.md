# Autograd slicing

`Variable.slice` keeps the selected rows in the computation graph. Backprop
routes gradients to those source positions and leaves all other positions at
zero.

Run from `~/.vmodules`:

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 v run ./vtl/examples/autograd_slice/main.v
```

Expected output:

```text
sliced: [1.0, 2.0, 5.0, 6.0]
source gradient: [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
```
