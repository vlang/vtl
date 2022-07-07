# Tutorial: Slicing

VTL supports slicing. It allows for selecting dimension subsets, whole dimension,
stepping (one out of 2 rows), reversing dimensions, counting from the end.

```v
import math
import vtl

const xs = [1, 2, 3, 4, 5]

const ys = [1, 2, 3, 4, 5]

mut vandermont := [][]int{}

for i, x in xs {
	row := []int{}
	vandermont << row
	for y in ys {
		vandermont[i] << int(math.pow(x, y))
	}
}

t := vtl.from_2d(vandermont)

println(t.shape) // [5, 5]

slice := t.slice_hilo([2, 3], [4, 5]) // [[16, 32], [81, 243]]

println(slice)
```
