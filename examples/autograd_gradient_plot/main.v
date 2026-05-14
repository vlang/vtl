module main

// autograd_gradient_plot: Visualize a function and its automatically
// computed gradient using VTL's autograd engine and vsl.plot.
//
// Computes f(x) = x^3 - 3x^2 + 2x and its derivative f'(x) = 3x^2 - 6x + 2
// across a range of values, demonstrating that VTL's autograd produces
// correct gradients at every point.
import vtl
import vtl.autograd
import vsl.plot

const n_points = 100
const x_min = -1.0
const x_max = 4.0

fn main() {
	// === Compute f(x) and f'(x) at each point using autograd ===
	mut x_values := []f64{len: n_points}
	mut y_values := []f64{len: n_points}
	mut grad_values := []f64{len: n_points}

	for i in 0 .. n_points {
		xi := x_min + (x_max - x_min) * f64(i) / f64(n_points - 1)
		x_values[i] = xi

		// Create a fresh context for each evaluation
		ctx := autograd.ctx[f64]()
		x := ctx.variable(vtl.from_1d([xi])!, requires_grad: true)

		// f(x) = x^3 - 3x^2 + 2x
		three := ctx.variable(vtl.from_1d([3.0])!)
		two := ctx.variable(vtl.from_1d([2.0])!)

		x_cubed := x.pow(three)!
		x_squared := x.pow(two)!
		three_x_sq := x_squared.mul_scalar(3.0)!
		two_x := x.mul_scalar(2.0)!

		// x^3 - 3x^2
		term1 := x_cubed.sub(three_x_sq)!
		// x^3 - 3x^2 + 2x
		mut result := term1.add(two_x)!

		y_values[i] = result.value.get([0])

		// Backpropagate to get df/dx
		result.backprop()!
		grad_values[i] = x.grad.get([0])
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
		name: 'f(x) = x\u00b3 - 3x\u00b2 + 2x'
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
		name: "f'(x) = 3x\u00b2 - 6x + 2 (autograd)"
	)
	// Zero line for reference
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
		title: 'VTL Autograd: Function and Gradient Visualization'
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

	// === Print critical points where f'(x) ≈ 0 ===
	println('Critical points (where gradient crosses zero):')
	for i in 1 .. n_points {
		if (grad_values[i - 1] >= 0 && grad_values[i] < 0)
			|| (grad_values[i - 1] <= 0 && grad_values[i] > 0) {
			// Linear interpolation to find approximate zero crossing
			t := grad_values[i - 1] / (grad_values[i - 1] - grad_values[i])
			x_cross := x_values[i - 1] + t * (x_values[i] - x_values[i - 1])
			println('  x ≈ ${x_cross:.4f}')
		}
	}
}
