# NumPy `.npy` round trip

This example writes a VTL tensor to a NumPy `.npy` file and reads it back.
Supported primitive numeric dtypes must match the requested V element type.
The reader accepts little-endian, big-endian, and Fortran-order arrays.

Run from `~/.vmodules`:

```bash
v run vtl/examples/npy_round_trip/main.v
```
