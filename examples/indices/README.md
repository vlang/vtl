# Dense coordinate indices

`indices` returns a dense coordinate tensor with one leading plane per input
dimension, following `numpy.indices` shape and row-major ordering.

Run from `~/.vmodules`:

```sh
VJOBS=2 v run ./vtl/examples/indices/main.v
```
