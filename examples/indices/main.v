import vtl

coordinates := vtl.indices([2, 3])!
println('coordinate tensor shape: ${coordinates.shape}')
println('row coordinates: ${coordinates.get([0, 1, 2])}')
println('column coordinates: ${coordinates.get([1, 1, 2])}')
