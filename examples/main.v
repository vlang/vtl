module main

import vtl

fn main() {
	sample := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [2, 4])

	mut a := sample.slice_hilo([0, 0], [2, 2])


	println(sample.data)
	println('sample = $sample\n')
        println(sample.shape)
        println(sample.strides)

	println('a = $a\n')
	println(a.data)
        println(a.shape)
        println(a.strides)

        for i in 0 .. a.size {
                println('i=$i, index=${a.nth_index(i)}, val=${a.get(a.nth_index(i))}')
        }
        println('\n\n')

        zeros := vtl.zeros<int>([2, 2])

        a.assign(zeros)

        println(sample.data)
	println('sample = $sample\n')

	println('a = $a\n')
	println(a.data)

}
