# Broadcasted selection with `choose`

Select values from two broadcast-compatible tensors using an integer choice tensor. The choice
tensor and candidates broadcast to one output shape.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/array_choose/main.v
```

## Notes

Use this when each output position chooses from a corresponding position in one of several
candidate tensors.
