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

values := vtl.from_1d([-2.0, 0.0, 3.0])!
println(values.sign())
println(values.heaviside(0.5))
