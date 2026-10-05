# Stable variance and standard deviation

This example shows population and sample variance using `ddof`, plus standard
deviation. The API returns `f64` for integer tensors and uses Welford's stable
online algorithm.

Run from `~/.vmodules`:

```bash
v run vtl/examples/stats_variance/main.v
```
