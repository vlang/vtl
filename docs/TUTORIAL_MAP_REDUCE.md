# Tutorial: Higher Order Functions (Map, Reduce and Fold)

VTL supports higher order functions. It allows for mapping, reducing and folding over vectors and matrices.

## map, nmap, apply, napply

The `map` function applies a function to each element of a tensor and returns a new tensor with the results.

```v
import vtl

mut t := vtl.from_1d([1, 2, 3, 4, 5])!

println(t)
// [1, 2, 3, 4, 5]

println(t.shape) // [5]

println('map: ')
map1 := t.map(fn (x int, idx []int) int { return x * 2 })

println(map1)
// [2, 4, 6, 8, 10]

println(map1.shape) // [5]
```

The `nmap` function applies a function to each element of a tensor and returns a new tensor with the results. The function receives the index of the element as a parameter.

```v
import vtl

mut t := vtl.from_1d([1, 2, 3, 4, 5])!

println(t)
// [1, 2, 3, 4, 5]

println(t.shape) // [5]

a := vtl.from_1d([2, 4, 6, 8, 10])!
b := vtl.from_1d([0, 1, 2, 3, 4])!

println('nmap: ')
nmap1 := t.nmap([a, b], fn (xs []int, idx []int) int { return xs[0] * xs[1] - xs[2] })!

println(nmap1)
// [2, 7, 16, 29, 46]

println(nmap1.shape) // [5]
```

The `apply` function applies a function to each element of a tensor. It is similar to `map` but it mutates the tensor in place.

```v
import vtl

mut t := vtl.from_1d([1, 2, 3, 4, 5])!

println(t)
// [1, 2, 3, 4, 5]

println(t.shape) // [5]

println('apply: ')
t.apply(fn (x int, idx []int) int { return x * 2 })

println(t)
// [2, 4, 6, 8, 10]

println(t.shape) // [5]
```

The `napply` function applies a function to each element of a tensor. It is similar to `nmap` but it mutates the tensor in place. The function receives the index of the element as a parameter.

```v
import vtl

mut t := vtl.from_1d([1, 2, 3, 4, 5])!

println(t)
// [1, 2, 3, 4, 5]

println(t.shape) // [5]

a := vtl.from_1d([2, 4, 6, 8, 10])!
b := vtl.from_1d([0, 1, 2, 3, 4])!

println('napply: ')
t.napply([a, b], fn (xs []int, idx []int) int { return xs[0] * xs[1] - xs[2] })!

println(t)
// [2, 7, 16, 29, 46]

println(t.shape) // [5]
```

## reduce, nreduce

The `reduce` function applies a function to each element of a tensor and returns a new tensor with the results. The function receives the index of the element as a parameter.

```v
import vtl

mut t := vtl.from_1d([1, 2, 3, 4, 5])!

println(t)
// [1, 2, 3, 4, 5]

println(t.shape) // [5]

println('reduce: ')
prod := t.reduce(1, fn (acc int, x int, idx []int) int { return acc * x })

println(prod) // 120
```

The `nreduce` function applies a function to each element of a tensor and returns a new tensor with the results. The function receives the index of the element as a parameter.

```v
import vtl

mut t := vtl.from_1d([1, 2, 3, 4, 5])!

println(t)
// [1, 2, 3, 4, 5]

println(t.shape) // [5]

a := vtl.from_1d([2, 4, 6, 8, 10])!

println('nreduce: ')
result := t.nreduce([a], 1, fn (acc int, xs []int, idx []int) int { return acc * xs[0] - xs[1] })!

println(result) // -530
```
