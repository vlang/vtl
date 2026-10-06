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

`random_seed` is a convenience wrapper around V's global generator. Reseeding
also changes the sequence observed by other code that uses V's global `rand`
module. Results are reproducible for the same VTL/V runtime and sequence of
random calls; they are not promised to stay bit-identical across RNG
implementation changes.

## Independent generators

Use `new_random_generator(seed)` when a data pipeline or experiment should own
its random stream. Each instance advances independently, so sampling in one
experiment does not change another experiment's sequence or V's global random
state:

```v
mut training_rng := vtl.new_random_generator(42)
mut validation_rng := vtl.new_random_generator(2026)
training_noise := training_rng.normal([128, 32], vtl.NormalTensorData{sigma: 0.1})!
validation_mask := validation_rng.bernoulli(0.8, [128])!
```

`uniform(minimum, maximum, shape)` produces `f64` samples in the half-open
range `[minimum, maximum)`. `normal(shape, params)` produces `f64` samples, and
`bernoulli(probability, shape)` produces boolean samples. These methods return
errors for invalid distribution parameters. Streams are reproducible with the
same V/VTL versions, seed, and sequence of calls; cross-version compatibility
is not guaranteed.

VTL currently provides uniform range, normal, Bernoulli, binomial, and
exponential tensor constructors. See [`rand.v`](../rand.v) for their API.
