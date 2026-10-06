# Numerical gradient

Estimate the derivative of sampled values along a selected tensor axis:

```bash
cd ~/.vmodules
systemd-run --user --scope --quiet --property=MemoryMax=768M --property=MemorySwapMax=0 \
	-- env VJOBS=2 v run ./vtl/examples/gradient/main.v
```

`stats.gradient_axis` preserves the input shape, accepts negative axes and
returns `f64` values. It uses centered differences inside the axis and
first-order one-sided differences at the two boundaries.

For non-uniform coordinates, use `stats.gradient_axis_with_coordinates` for
first-order boundaries, or `stats.gradient_axis_with_coordinates_edge_order`
to select `edge_order` 1 or 2. The example demonstrates exact second-order
derivatives for samples of a quadratic function.
