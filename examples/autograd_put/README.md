# Autograd indexed replacement

`Variable.put_along_axis`, `Variable.put`, and `Variable.put_with_mode` return
copied tensors with indexed replacements. For duplicate destinations, the
final update wins and only that update gets a gradient. Flat `put` repeats
short update tensors, and repeated writes to a destination update only the
gradient slot for the value that won.

Run from `~/.vmodules`:

```bash
VJOBS=2 v run ./vtl/examples/autograd_put/main.v
```

Expected output:

```text
put: [10.0, 200.0, 300.0]
source gradient: [2.0, 0.0, 0.0]
updates gradient: [0.0, 3.0, 5.0]
flat put: [10.0, 200.0, 30.0, 100.0]
flat source gradient: [2.0, 0.0, 5.0, 0.0]
flat updates gradient: [7.0, 3.0]
```
