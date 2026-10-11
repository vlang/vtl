# Priority-based tensor selection

Use `vtl.select` to choose among several broadcast-compatible tensors using
boolean masks. The first matching condition wins, and `default` fills positions
where every condition is false, matching NumPy's `select` priority behavior.

## Run

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/array_select/main.v
```

## Notes

All conditions and choices broadcast to one common output shape. This API keeps
the choice and default values at one V element type; mixed-dtype promotion is
not implemented.
