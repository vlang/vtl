module main

import vtl

fn main() {
	mut t := vtl.seq<f64>(10)
	println(t.str())
	t = vtl.from_array<f64>([1.0, 2, 3, 4, 5], [5])
	println(t.copy(.colmajor))
	println(t.view())
	m := vtl.from_array<f64>([1.0, 2.0, 3.0], [3, 1])
	println(m)
	b := m.broadcast_to([3, 3])
	println(b)
	expected := vtl.from_array<f64>([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0], [3, 3])
	println(expected)
	mut iter := expected.iterator()
	for {
		val, _ := iter.next() or { break }
		println(val)
	}
	println('')
	println('')
	println('')
	println('')
	println('')

	res := vtl.eye<f64>(2, 4, 0)
	expected2 := vtl.from_array<f64>([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [2, 4])
	println(res)
	println(expected2)
	println(res.equal(expected2))

	a1 := vtl.from_array([0.0, 1, 2, 3, 4, 5, 6, 7, 8], [3, 3])
	slice := a1.slice([0])
	expected3 := vtl.from_array([0.0, 1, 2], [3])
	println(slice.equal(expected3))
	mat := vtl.from_2d([[2.0, 3, 4], [1.0, 2, 3]])
	println(mat.str())
	println(vtl.sin<f64>(a1).str())
	println(vtl.pow<f64>(a1, a1).str())

	// @todo: FIX THIS
	a2 := vtl.ones<f64>([2, 2])
	b2 := vtl.zeros<f64>([2, 2])
	result_ := vtl.hstack<f64>([a2, b2])
	expected_ := vtl.from_array<f64>([1.0, 1, 0, 0, 1, 1, 0, 0], [2, 4])
	println(result_.str())
	println(expected_.str())
	println(result_.equal(expected_))

	a3 := vtl.ones<f64>([3])
	mut b3 := vtl.zeros<f64>([2])
	mut result3_ := vtl.concatenate<f64>([a3, b3], axis: 0)
	expected3_ := vtl.from_1d<f64>([1.0, 1, 1, 0, 0])
	println(result3_.equal(expected3_))
	b4 := vtl.ones<f64>([2])
	b3.assign(b4)
	println(b3.str())
}
