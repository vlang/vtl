# Broadcasted tensor clipping

`clip_tensor` clamps each value between corresponding lower and upper bounds.
The bounds use NumPy-style broadcasting, and the operation writes one output
tensor in a single pass.

Run from the V modules directory:

```sh
cd ~/.vmodules
VJOBS=2 v run ./vtl/examples/clip/main.v
```
