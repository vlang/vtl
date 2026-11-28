# CPU profiler

`vtl.profiler` is an opt-in timing and memory-accounting module for
[#741](https://github.com/vlang/vtl/issues/741). It reports operation timing
and peak/active CPU memory for bytes that callers explicitly attribute to
labels. It does not intercept VTL allocations automatically, so overhead is
paid only where instrumentation is inserted.

```v
mut p := profiler.new(true)
p.allocate('weights', 4096)!
value := profiler.timed[f64](mut p, 'forward', fn () !f64 {
	return 1.0
})!
p.release('weights', 4096)!
println(p.report())
```

- `allocate(label, bytes)` / `release(label, bytes)` track live buffers per
  label. Over-release and negative sizes are rejected.
- `peak_active_bytes()` is the maximum total live bytes; `active_bytes(label)`
  and `active_total_bytes()` report current usage.
- `timed(label, op)` records elapsed nanoseconds and still records the event
  when `op` fails, then propagates the error.
- `tensor_bytes[T](shape)` computes dense byte size with zero-dimension and
  overflow handling.
- `new(false)` keeps call sites in place while recording nothing.

Run the focused tests with:

```sh
v test tests/profiler/profiler_test.v
```

Reusable-buffer planning, accelerator memory attribution, operation-level
instrumentation inside VTL kernels, and memory-budget examples remain tracked
in [#741](https://github.com/vlang/vtl/issues/741).
