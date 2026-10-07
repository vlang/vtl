# Autograd indexed replacement

`Variable.put_along_axis` returns a copied tensor with indexed replacements.
For duplicate destinations, the final update wins and only that update gets a
gradient.

Run from `~/.vmodules`:

```bash
VJOBS=2 v run ./vtl/examples/autograd_put/main.v
```

Expected output:

```text
put: [10.0, 200.0, 300.0]
source gradient: [2.0, 0.0, 0.0]
updates gradient: [0.0, 3.0, 5.0]
```
