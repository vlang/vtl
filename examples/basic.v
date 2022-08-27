import vtl

a := vtl.from_1d([1, 2, 3, 4]) or { panic(err) }
b := vtl.from_1d([0, 1, 2, 3]) or { panic(err) }

c := a.add<int>(b) or { panic(err) }

print(c)
