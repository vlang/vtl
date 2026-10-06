# Random numbers and reproducibility

VTL random tensor constructors use V's global random generator. Call
`vtl.random_seed(seed)` before generating tensors or initializing a model to
repeat the same sequence of random values:

```v
import vtl

vtl.random_seed(42)
first := vtl.random[f64](0.0, 1.0, [4], vtl.TensorData{})

vtl.random_seed(42)
again := vtl.random[f64](0.0, 1.0, [4], vtl.TensorData{})
assert first.array_equal(again)
```

The seed also affects random model parameter initialization and stochastic
layers that use V's default generator. Reset it before constructing a model or
running an experiment whose random sequence must be repeatable.

`random_seed` is a convenience wrapper around V's global generator. VTL does
not yet provide independent generator objects, so reseeding also changes the
sequence observed by other code that uses V's global `rand` module. Results
are reproducible for the same VTL/V runtime and sequence of random calls; they
are not promised to stay bit-identical across RNG implementation changes.

VTL currently provides uniform range, normal, Bernoulli, binomial, and
exponential tensor constructors. See [`rand.v`](../rand.v) for their API.
