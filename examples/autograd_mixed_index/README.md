# Autograd mixed indexing

This example gathers values using paired coordinate arrays and backpropagates
through a repeated coordinate. The repeated selection accumulates in the
source gradient.

Run from `~/.vmodules`:

```bash
VJOBS=2 v run ./vtl/examples/autograd_mixed_index/main.v
```

Expected output:

```text
selected: [5.0, 0.0, 5.0]
source gradient: [1.0, 0.0, 0.0, 0.0, 0.0, 2.0]
```
