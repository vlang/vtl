# Mixed-precision helpers

`vtl.amp` provides backend-independent gradient-scaling primitives for
[#742](https://github.com/vlang/vtl/issues/742). It does not change tensor
dtype or dispatch kernels; it implements the scaling state machine that sits
around a low-precision training step.

```v
mut scaler := amp.new(amp.GradScalerConfig{init_scale: 65536.0})!
scaled_loss := scaler.scale_loss(loss)
// backprop scaled_loss, collect gradients
ok := scaler.unscale(mut gradients)!
if ok {
	optimizer.update()!
}
scaler.update(!ok)
```

- `scale_loss` multiplies a loss by the current scale; `unscale` divides
  gradients in place and returns false if any value is non-finite.
- Finite gradients are still unscaled when another gradient overflows;
  non-finite values are left untouched.
- `update` backs off on overflow and grows after `growth_interval` consecutive
  good steps, using the configured factors.
- `state()` / `from_state()` checkpoint the scale and good-step counter.
- `enabled: false` makes every operation a no-op with scale 1.

Supported gradient dtypes are `f32` and `f64`. Autocast policies, device-aware
scaling, and distributed synchronization remain tracked in #742.

Run the focused tests with:

```sh
v test tests/amp/grad_scaler_test.v
```
