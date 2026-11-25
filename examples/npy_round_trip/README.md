# NumPy `.npy` round trip

This example writes a VTL tensor to a NumPy `.npy` file, reads it back, and
retrieves one row through a read-only memory mapping.
Supported primitive numeric dtypes must match the requested V element type.
The reader accepts little-endian, big-endian, and Fortran-order arrays.
Mapped access decodes the requested row range into an owned tensor; call
`close()` when finished and do not modify or truncate the file while mapped.

Run from `~/.vmodules`:

```bash
v run vtl/examples/npy_round_trip/main.v
```
