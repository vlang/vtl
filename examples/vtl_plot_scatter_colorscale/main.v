module main

import vsl.plot
import vtl

fn main() {
	y := [
		0.0,
		1,
		3,
		1,
		0,
		-1,
		-3,
		-1,
		0,
		1,
		3,
		1,
		0,
	]
	x := vtl.seq[f64](y.len)

	mut plt := plot.Plot.new()
	plt.scatter(
		x:          x.to_array()
		y:          y
		mode:       'lines+markers'
		colorscale: 'smoker'
		marker:     plot.Marker{
			size: []f64{len: x.size, init: 10.0}
		}
	)
	plt.layout(
		title: 'Scatter plot example'
	)
	plt.show()!
}
