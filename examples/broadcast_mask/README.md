# Broadcastable tensor masks

Use a one-dimensional mask to select columns and a column-shaped mask to fill whole rows of a matrix.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/broadcast_mask/main.v
```

## Notes

Mask dimensions follow tensor broadcasting rules; `masked_select` returns selected values in logical order and `masked_fill` returns a filled copy.
