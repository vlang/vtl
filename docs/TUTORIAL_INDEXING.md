# Indexing, gather, and scatter

`take` and `take_along_axis` gather copies from a tensor. Negative axes and
indices count from the end. See [slicing](./TUTORIAL_SLICING.md) when a view of
a range is more appropriate.

`put_along_axis` writes values into a mutable tensor. `scatter_add` adds values
to the selected locations; duplicate indices accumulate in row-major order.
Both operations require the index tensor and values tensor to have matching
shapes and ranks. Dimensions outside the selected axis may be smaller than the
target, and negative indices are supported.

```v
import vtl

mut values := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
indices := vtl.from_array([1, 1, 0, 0], [2, 2])!
updates := vtl.from_array([10, 20, 30, 40], [2, 2])!

values.scatter_add(indices, updates, 1)!
assert values.to_array() == [1, 32, 3, 74, 5, 6]

values.put_along_axis(indices, updates, 1)!
assert values.to_array() == [1, 20, 3, 40, 5, 6]
```

`put_along_axis` applies repeated writes in row-major order, so the last value
for a repeated index wins. Both update operations validate all indices before
mutating the target. Indexed writes currently operate on tensors directly and
are not autograd operations.
