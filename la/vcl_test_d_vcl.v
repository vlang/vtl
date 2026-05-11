module la

import vtl

fn test_matmul_vcl_identity() {
	// 2x2 identity * identity = identity
	a := vtl.from_2d[f64]([[1.0, 0.0], [0.0, 1.0]]) or { panic(err) }
	b := vtl.from_2d[f64]([[1.0, 0.0], [0.0, 1.0]]) or { panic(err) }
	c := matmul_vcl[f64](a, b) or { panic('matmul_vcl failed: ${err}') }
	ca := c.to_array()
	assert ca[0] == 1.0
	assert ca[1] == 0.0
	assert ca[2] == 0.0
	assert ca[3] == 1.0
}

fn test_matmul_vcl_basic() {
	// [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
	a := vtl.from_2d[f64]([[1.0, 2.0], [3.0, 4.0]]) or { panic(err) }
	b := vtl.from_2d[f64]([[5.0, 6.0], [7.0, 8.0]]) or { panic(err) }
	c := matmul_vcl[f64](a, b) or { panic('matmul_vcl failed: ${err}') }
	ca := c.to_array()
	assert ca[0] == 19.0
	assert ca[1] == 22.0
	assert ca[2] == 43.0
	assert ca[3] == 50.0
}

fn test_matmul_vcl_f32_basic() {
	a := vtl.from_2d[f32]([[1.0, 2.0], [3.0, 4.0]]) or { panic(err) }
	b := vtl.from_2d[f32]([[5.0, 6.0], [7.0, 8.0]]) or { panic(err) }
	c := matmul_vcl_f32(a, b) or { panic('matmul_vcl_f32 failed: ${err}') }
	ca := c.to_array()
	assert ca[0] == f32(19.0)
	assert ca[1] == f32(22.0)
	assert ca[2] == f32(43.0)
	assert ca[3] == f32(50.0)
}

fn test_matmul_vcl_rect() {
	// [2x3] * [3x2]
	a := vtl.from_2d[f64]([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) or { panic(err) }
	b := vtl.from_2d[f64]([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]) or { panic(err) }
	c := matmul_vcl[f64](a, b) or { panic('matmul_vcl rect failed: ${err}') }
	assert c.shape == [2, 2]
	ca := c.to_array()
	// row0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
	// row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
	assert ca[0] == 58.0
	assert ca[1] == 64.0
	assert ca[2] == 139.0
	assert ca[3] == 154.0
}
