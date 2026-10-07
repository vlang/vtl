import vtl

fn main() {
	volume := vtl.from_array([0.0, 5.0, 3.0, 7.0, 4.0, 2.0, 8.0, 1.0], [2, 2, 2]) or {
		panic(err)
	}
	maxima := volume.max_axes([0, -1], false) or { panic(err) }
	minima := volume.min_axes([0, 2], true) or { panic(err) }
	println('maximum over outer axes: ${maxima.to_array()} with shape ${maxima.shape}')
	println('minimum with dimensions retained: ${minima.to_array()} with shape ${minima.shape}')
}
