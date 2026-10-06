module core

import vtl

fn test_meshgrid_n_supports_xy_and_ij_indexing() ! {
	x := vtl.from_1d([10, 20])!
	y := vtl.from_1d([1, 2, 3])!
	z := vtl.from_1d([4, 5, 6, 7])!

	xy := vtl.meshgrid_n[int]([x, y, z], .xy)!
	assert xy.len == 3
	assert xy[0].shape == [3, 2, 4]
	assert xy[0].get([2, 1, 3]) == 20
	assert xy[1].get([2, 1, 3]) == 3
	assert xy[2].get([2, 1, 3]) == 7

	ij := vtl.meshgrid_n[int]([x, y, z], .ij)!
	assert ij[0].shape == [2, 3, 4]
	assert ij[0].get([1, 2, 3]) == 20
	assert ij[1].get([1, 2, 3]) == 3
	assert ij[2].get([1, 2, 3]) == 7

	empty := vtl.meshgrid_n[int]([vtl.from_1d([]int{})!, z], .xy)!
	assert empty[0].shape == [4, 0]
	assert empty[1].shape == [4, 0]
}

fn test_meshgrid_n_validates_inputs() ! {
	if _ := vtl.meshgrid_n[int]([], .ij) {
		assert false, 'meshgrid_n must reject an empty vector list'
	}
	x := vtl.from_2d([[1, 2]])!
	if _ := vtl.meshgrid_n[int]([x], .ij) {
		assert false, 'meshgrid_n must reject non-vector inputs'
	}
}
