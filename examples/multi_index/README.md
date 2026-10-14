# Multi-index conversion

Convert broadcast coordinate tensors to flat indices and recover coordinates
in either C or Fortran order. The example also shows NumPy-style wrapping for
out-of-range coordinates.

Run from `~/.vmodules`:

```sh
VJOBS=2 v run ./vtl/examples/multi_index/main.v
```
