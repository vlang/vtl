import vtl

a := vtl.from_1d([1, 2, 3, 4])!
b := vtl.from_1d([0, 1, 2, 3])!

mut c := a.add(b)!

println(c)

c.apply(fn (x int, i []int) int {
	return x * 2
})

println(c)

d := c.map(fn (x int, i []int) int {
	return x * 2
})

println(d)
