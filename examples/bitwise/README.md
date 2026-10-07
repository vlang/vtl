# Bitwise tensor operations

Use integer tensors for elementwise bitwise AND, OR, XOR, and inversion.
Binary operators broadcast tensor shapes; left and right shifts take a scalar
bit count and reject counts outside the element type's width.

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/bitwise/main.v
```
