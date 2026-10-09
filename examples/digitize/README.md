# Digitize values into bins

Map measurements to monotonically increasing bin intervals with `digitize` and compare the result
to right-sided `searchsorted` positions.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/digitize/main.v
```

## Notes

The `right` argument controls which side receives values equal to an edge; see the API docs for
boundary behavior.
