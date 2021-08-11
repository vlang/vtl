module main

import vtl

fn main() {
	mut t := vtl.seq<f64>(10)
	println(t)
	t = vtl.from_array<f64>([1., 2, 3, 4, 5], [5])
	println(t.copy(.colmajor))
	println(t.view())
	m := vtl.from_array<f64>([1., 2., 3.], [3, 1])
	println(m)
	b := m.broadcast_to([3, 3])
	println(b)
	expected := vtl.from_array<f64>([1., 1., 1., 2., 2., 2., 3., 3., 3.], [3, 3])
	println(expected)
	mut iter := expected.iterator()
	for _ in 0 .. expected.size {
		println(iter.next() ?)
	}
	println('')
	println('')
	println('')
	println('')
	println('')

	res := vtl.eye<f64>(2, 4, 0)
	expected2 := vtl.from_array<f64>([1.0, 0., 0., 0., 0., 1., 0., 0.], [2, 4])
	println(res)
	println(expected2)
	println(res.equal(expected2))
}
