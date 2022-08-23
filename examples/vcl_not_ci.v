import vtl

t := vtl.from_1d([1, 2, 3, 4, 5, 6])?
cl := t.vcl()?
println(cl)
