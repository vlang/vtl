import vtl

x := vtl.from_1d([1.0, 2.0, 3.0])!
descending := vtl.vander(x)!
increasing := vtl.vander(x, increasing: true)!

println('Descending powers:')
println(descending)
println('Increasing powers:')
println(increasing)
