# Autograd gather

This example gathers columns from a tensor and backpropagates through repeated
indices. Each repeated selection contributes to the source gradient.

From `~/.vmodules`, run:

```bash
VJOBS=2 v run ./vtl/examples/autograd_gather/main.v
```

The output is:

```text
selected: [20.0, 20.0, 30.0]
source gradient: [0.0, 2.0, 1.0]
```
