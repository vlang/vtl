import vtl

a := vtl.from_1d([1, 2, 3, 4]) or { panic(err) }
b := vtl.from_1d([0, 1, 2, 3]) or { panic(err) }

c := vtl.add<int>(a, b) or { panic(err) }

print(c)
