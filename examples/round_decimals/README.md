# Round to decimal places

`Tensor.round_decimals` rounds floating-point values to positive or negative
decimal places using ties-to-even, like NumPy's `round`.

## Run

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/round_decimals/main.v
```

## Notes

Floating-point rounding follows binary representation, so decimal values near a
halfway boundary can reflect the same representation effects as NumPy. Integer
tensors are copied unchanged. If a positive decimal position would overflow an
intermediate scale, VTL preserves finite inputs; an unrepresentably large
negative position rounds finite values to zero instead of producing invalid
intermediate values.
