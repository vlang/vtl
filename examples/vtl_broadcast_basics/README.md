# vtl_broadcast_basics

Demonstrates implicit broadcasting in VTL using:

- column vector `[4, 1]`
- row vector `[1, 3]`
- scalar broadcasting

## Run

```sh
v run main.v
```

## Expected behavior

- `col + row` produces a `[4, 3]` tensor
- `grid + scalar` shifts all values by the scalar
