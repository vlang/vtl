# NumPy complex128 `.npy` round trip

This example saves and reloads a VTL complex tensor using NumPy's `complex128`
`.npy` dtype. The reader and writer preserve both real and imaginary components.

Run from `~/.vmodules`:

```sh
VJOBS=2 v run ./vtl/examples/npy_complex128/main.v
```
