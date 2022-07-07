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

println(t)
// [[   1,    1,    1,    1,    1],
// [   2,    4,    8,   16,   32],
// [   3,    9,   27,   81,  243],
// [   4,   16,   64,  256, 1024],
// [   5,   25,  125,  625, 3125]]

println(t.shape) // [5, 5]

println('slice: ')
slice1 := t.slice_hilo([2, 3], [4, 5])

println(slice1)
// [[16, 32], [81, 243]]

println(slice1.shape) // [2, 2]

slice2 := t.slice_hilo([3], [])

println('span slice: ')
println(slice2)
// [[   4,   16,   64,  256, 1024],
// [   5,   25,  125,  625, 3125]]

println(slice2.shape) // [2, 5]

slice3 := t.slice_hilo([], [-2])

println('slice until: ')
println(slice3)
// [[  1,   1,   1,   1,   1],
// [  2,   4,   8,  16,  32],
// [  3,   9,  27,  81, 243]]

println(slice3.shape) // [3, 5]
```
