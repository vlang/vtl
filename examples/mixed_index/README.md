# Mixed tensor indexing

Combine broadcasted integer coordinate arrays with scalar indices and
Python-style slices. Coordinate arrays with a slice between them place their
broadcast dimensions at the front of the result, matching NumPy. Basic
indexing returns a view; coordinate indexing returns a copy. Negative-step
slices currently return a copy because VTL cannot safely represent
negative-stride views.

Run from `~/.vmodules`:

```bash
VJOBS=2 v run ./vtl/examples/mixed_index/main.v
```
