# Autograd scatter-add

`Variable.scatter_add` creates an updated copy while recording an autograd
operation. This example uses repeated destinations and a non-uniform gradient
from a weighted downstream operation.

From `~/.vmodules`, run:

```bash
VJOBS=2 v run ./vtl/examples/autograd_scatter/main.v
```

The output is:

```text
scattered: [1.0, 32.0, 33.0]
source gradient: [2.0, 3.0, 5.0]
updates gradient: [3.0, 3.0, 5.0]
```
