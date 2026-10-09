# Average Pooling and Backpropagation

This example applies a 2×2 average-pooling window to a 3×3 input, then
backpropagates a unit gradient through the pooling layer. It prints the pooled
values and the accumulated input gradients, including the overlaps between
neighboring windows.

## Run

Run from `~/.vmodules` so V resolves the `vtl` module at `./vtl`:

```sh
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 env VJOBS=2 v run ./vtl/examples/nn_average_pooling/main.v
```

Expected values:

```text
Output shape: [1, 1, 2, 2]
Output values: [3.0, 4.0, 6.0, 7.0]
Input gradient: [0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]
```
