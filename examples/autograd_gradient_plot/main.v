module main

// autograd_gradient_plot: Visualize a function and its automatically
// computed gradient using VTL's autograd engine and vsl.plot.
//
// Computes f(x) = sin(x) and its derivative f'(x) = cos(x)
// across a range of values, demonstrating that VTL's autograd produces
// correct gradients at every point.
import math
import vtl
import vtl.autograd
import vsl.plot

const n_points = 100
const x_min = -math.pi * 2
const x_max = math.pi * 2

fn main() {
	// === Compute f(x) = sin(x) and f'(x) at each point using autograd ===
	mut x_values := []f64{len: n_points}
	mut y_values := []f64{len: n_points}
	mut grad_values := []f64{len: n_points}
	mut expected_grad := []f64{len: n_points}

	for i in 0 .. n_points {
		xi := x_min + (x_max - x_min) * f64(i) / f64(n_points - 1)
		x_values[i] = xi

		// Create a fresh context for each evaluation
		ctx := autograd.ctx[f64]()
		x := ctx.variable(vtl.from_1d([xi])!, requires_grad: true)

		// f(x) = sin(x) — VTL's autograd supports sin natively
		mut result := x.sin()!

		y_values[i] = result.value.get([0])

		// Backpropagate to get df/dx = cos(x)
		result.backprop()!
		grad_values[i] = x.grad.get([0])
		expected_grad[i] = math.cos(xi)
	}

	// === Plot f(x) and f'(x) ===
	mut plt := plot.Plot.new()

	plt.scatter(
		x:    x_values
		y:    y_values
		mode: 'lines'
		line: plot.Line{
			color: '#2196F3'
			width: 2.5
		}
		name: 'f(x) = sin(x)'
	)
	plt.scatter(
		x:    x_values
		y:    grad_values
		mode: 'lines'
		line: plot.Line{
			color: '#FF5722'
			width: 2.5
			dash:  'dash'
		}
		name: "f'(x) via autograd"
	)
	plt.scatter(
		x:    x_values
		y:    expected_grad
		mode: 'lines'
		line: plot.Line{
			color: '#4CAF50'
			width: 1.5
			dash:  'dot'
		}
		name: 'cos(x) (expected)'
	)
	// Zero line
	plt.scatter(
		x:    [x_min, x_max]
		y:    [0.0, 0.0]
		mode: 'lines'
		line: plot.Line{
			color: '#9E9E9E'
			width: 1.0
			dash:  'dot'
		}
		name: 'y = 0'
	)

	plt.layout(
		title: 'VTL Autograd: sin(x) and its Gradient cos(x)'
		xaxis: plot.Axis{
			title: plot.AxisTitle{
				text: 'x'
			}
		}
		yaxis: plot.Axis{
			title: plot.AxisTitle{
				text: 'y'
			}
		}
	)
	plt.show()!

	// === Verify autograd accuracy ===
	println('Autograd accuracy check:')
	println('  x         | autograd    | cos(x)      | error')
	println('  ----------|-------------|-------------|--------')
	check_points := [-math.pi, -math.pi / 2, 0.0, math.pi / 2, math.pi]
	for xv in check_points {
		ctx := autograd.ctx[f64]()
		x := ctx.variable(vtl.from_1d([xv])!, requires_grad: true)
		mut r := x.sin()!
		r.backprop()!
		ag := x.grad.get([0])
		expected := math.cos(xv)
		err := math.abs(ag - expected)
		println('  ${xv:9.4f} | ${ag:11.8f} | ${expected:11.8f} | ${err:.2e}')
	}
}
