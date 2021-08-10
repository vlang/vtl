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
}
