# Autograd scalar losses

Use `Variable.sum()` or `Variable.mean()` to reduce tensor values to a
one-element loss. Backpropagation distributes the gradient across the source
tensor; the mean scales each value by the reciprocal of the element count.

Run from `~/.vmodules`:

```bash
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- env VJOBS=2 \
	v run ./vtl/examples/autograd_scalar_loss/main.v
```

Expected output:

```text
sum loss: [13.0]
sum gradient: [4.0, 6.0]
mean loss: [6.5]
mean gradient: [2.0, 3.0]
```
