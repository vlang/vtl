# vtl_basic_usage

Shows the core VTL tensor operations: creation, element-wise arithmetic,
in-place mutation with `apply`, and out-of-place transformation with `map`.

## What it demonstrates

- Creating 1-D tensors from arrays with `vtl.from_1d`
- Element-wise addition with `.add()`
- In-place mutation with `.apply()` (modifies the tensor in place)
- Out-of-place transformation with `.map()` (returns a new tensor)

## How to run

```sh
v run main.v
```

## Expected output

```
[1, 3, 5, 7]       <- a + b
[2, 6, 10, 14]     <- apply(*2) in place
[4, 12, 20, 28]    <- map(*2), new tensor
```

## Key API

| Function / method | Description |
|-------------------|-------------|
| `vtl.from_1d(arr)` | Create a 1-D tensor from a V array |
| `a.add(b)` | Element-wise addition (returns new tensor) |
| `t.apply(fn)` | Mutate each element in place |
| `t.map(fn)` | Transform each element, return new tensor |

## Related examples

- [`vtl_vandermont`](../vtl_vandermont/) — 2-D tensor creation and slicing
- [`autograd_backprop`](../autograd_backprop/) — automatic differentiation
