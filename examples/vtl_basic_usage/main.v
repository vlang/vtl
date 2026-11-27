import vtl

left := vtl.from_1d([1, 2, 3, 4])!
right := vtl.from_1d([0, 1, 2, 3])!

mut sum := left.add(right)!

println(sum)

sum.apply(fn (x int, i []int) int {
	return x * 2
})

println(sum)

twice := sum.map(fn (x int, i []int) int {
	return x * 2
})

println(twice)

values := vtl.from_1d([-2.0, 0.0, 3.0])!
println(values.sign())
println(values.heaviside(0.5))
