module main

import vtl

fn main() {
	cold := vtl.from_2d([[true], [false]])!
	warm := vtl.from_2d([[false, true, false]])!
	mut conditions := []&vtl.Tensor[bool]{cap: 2}
	conditions << cold
	conditions << warm
	first_choice := vtl.from_2d([[0], [0]])!
	second_choice := vtl.from_2d([[1, 1, 1]])!
	mut choices := []&vtl.Tensor[int]{cap: 2}
	choices << first_choice
	choices << second_choice
	labels := vtl.select(conditions, choices, 2)!
	println('Labels (0=cold, 1=warm, 2=other):')
	println(labels.to_array())
}
