# Multidimensional gather

`take_nd` inserts the complete index tensor shape in place of the selected
axis. `take_flat` gathers from the logical row-major flattening and preserves
the index tensor shape.

Run from the V modules directory:

```sh
cd ~/.vmodules
VJOBS=2 v run ./vtl/examples/take_nd/main.v
```
