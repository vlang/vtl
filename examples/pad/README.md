# Pad tensor boundaries

Compare reflection, symmetric, and constant padding on a vector, including exact edge behavior
checked with assertions.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/pad/main.v
```

## Notes

`reflect` excludes the edge value from the reflected region; `symmetric` includes it. See the
[slicing tutorial](../../docs/TUTORIAL_SLICING.md).
