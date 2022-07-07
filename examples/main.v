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

println(t.shape)

slice := t.slice_hilo([1, 3], [3, 5])

println(slice)
