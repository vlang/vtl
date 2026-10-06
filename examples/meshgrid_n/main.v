module main

import vtl

fn main() {
	x := vtl.from_1d([1, 2])!
	y := vtl.from_1d([10, 20, 30])!
	z := vtl.from_1d([4, 5])!
	grids := vtl.meshgrid_n[int]([x, y, z], .xy)!
	println('Grid shape: ${grids[0].shape}')
	println('X at [2, 1, 1]: ${grids[0].get([2, 1, 1])}')
	println('Y at [2, 1, 1]: ${grids[1].get([2, 1, 1])}')
	println('Z at [2, 1, 1]: ${grids[2].get([2, 1, 1])}')
}
