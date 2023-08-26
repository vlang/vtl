import vtl

mut t := vtl.from_1d([1, 2, 3, 4, 5])!

println(t)
// [1, 2, 3, 4, 5]

println(t.shape) // [5]

a := vtl.from_1d([2, 4, 6, 8, 10])!

println('nreduce: ')
result := t.nreduce([a], 1, fn (acc int, xs []int, idx []int) int {
	return acc * xs[0] - xs[1]
})!

println(result) // -530
