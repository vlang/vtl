module main

import vtl
import vtl.datasets

fn main() {
	iris := datasets.load_iris() or { panic(err) }

	assert iris.features.shape == [150, 4]
	assert iris.labels.shape == [150]

	println('features shape: ${iris.features.shape}')
	println('labels shape: ${iris.labels.shape}')

	first_row := iris.features.slice([0, 1], [0, 4]) or { panic(err) }
	println('first feature row: ${first_row}')

	// Class distribution summary
	for class_id in [0, 1, 2] {
		class_mask := iris.labels.equal(vtl.tensor(class_id, [150])) or { panic(err) }
		count := class_mask.to_array[bool]().filter(it).len
		println('class ${class_id} count: ${count}')
	}
}
