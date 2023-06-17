import vtl

a := vtl.from_1d([1, 2, 3, 4])!
b := vtl.from_1d([0, 1, 2, 3])!

c := a.add(b)!

println(c)
