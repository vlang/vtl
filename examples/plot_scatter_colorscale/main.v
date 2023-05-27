module main

import vsl.plot
import vsl.util
import vtl

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

mut plt := plot.new_plot()
plt.add_trace(
	trace_type: .scatter
	x: x.to_array()
	y: y
	mode: 'lines+markers'
	colorscale: 'smoker'
	marker: plot.Marker{
		size: []f64{len: x.size, init: 10.0}
	}
)
plt.set_layout(
	title: 'Scatter plot example'
)
plt.show()!
