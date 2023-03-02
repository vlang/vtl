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

mut t := vtl.from_2d(vandermont)!

println(t)
// [[   1,    1,    1,    1,    1],
// [   2,    4,    8,   16,   32],
// [   3,    9,   27,   81,  243],
// [   4,   16,   64,  256, 1024],
// [   5,   25,  125,  625, 3125]]

println(t.shape) // [5, 5]

println('slice: ')
mut slice1 := t.slice_hilo([1, 3], [3, 5])!

println(slice1)
// [[16, 32], [81, 243]]

println(slice1.shape) // [2, 2]

slice2 := t.slice_hilo([3], [])!

println('span slice: ')
println(slice2)
// [[   4,   16,   64,  256, 1024],
// [   5,   25,  125,  625, 3125]]

println(slice2.shape) // [2, 5]

slice3 := t.slice_hilo([], [-2])!

println('slice until: ')
println(slice3)
// [[  1,   1,   1,   1,   1],
// [  2,   4,   8,  16,  32],
// [  3,   9,  27,  81, 243]]

println(slice3.shape) // [3, 5]

t999 := vtl.tensor(999, [2, 2])

slice1.assign(t999)?

println('assign: ')
println(t)
// [[   1,    1,    1,    1,    1],
// [   2,    4,    8,  999,  999],
// [   3,    9,   27,  999,  999],
// [   4,   16,   64,  256, 1024],
// [   5,   25,  125,  625, 3125]]
