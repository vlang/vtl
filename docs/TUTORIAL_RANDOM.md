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
import vtl

mut training_rng := vtl.new_random_generator(42)
mut validation_rng := vtl.new_random_generator(2026)
training_features := training_rng.normal([128, 32], vtl.NormalTensorData{ sigma: 0.1 })!
validation_mask := validation_rng.bernoulli(0.8, [128])!
waiting_times := validation_rng.geometric(0.1, [128])!
training_indices := vtl.from_1d([0, 1, 2, 3, 4, 5, 6, 7])!
batch_indices := training_rng.choice[int](training_indices, 4, false)!
sampling_weights := vtl.from_1d([1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0])!
weighted_indices := training_rng.choice_weighted[int](training_indices, sampling_weights, 4, true)!
epoch_order := training_rng.permutation(8)!
shuffled_rows := training_rng.permutation_tensor(training_features)!
positive_noise := training_rng.gamma(2.0, 0.5, [128])!
beta_samples := training_rng.beta(2.0, 5.0, [128])!
positive_scales := training_rng.lognormal(0.0, 0.25, [128])!
event_counts := training_rng.binomial(12, 0.25, [128])!
waiting_durations := validation_rng.exponential(0.5, [128])!
```

`uniform(minimum, maximum, shape)` produces `f64` samples in the half-open
range `[minimum, maximum)`. `normal(shape, params)` produces `f64` samples, and
`bernoulli(probability, shape)` produces boolean samples. `geometric(probability,
shape)` returns the positive number of Bernoulli trials up to the first success.
`choice(population, size, replace)` samples from the population in flattened
logical order and can enforce unique sampled positions. `permutation(size)`
returns every integer from zero to `size - 1` exactly once in a seeded order.
`choice_weighted(population, weights, size, replace)` samples in proportion to
finite non-negative weights and can sample without replacement.
`permutation_tensor(tensor)` returns a copy with complete rows shuffled along
the first axis, preserving all feature values in each row.
`gamma(alpha, scale, shape)` samples positive values from a Gamma distribution
on the same independent stream. `lognormal(mean, sigma, shape)` exponentiates
samples from a normal distribution; a zero `sigma` returns the constant
`exp(mean)`. `beta(alpha, beta, shape)` samples values in
`[0, 1]` using the same seeded stream. `binomial(trials, probability, shape)`
returns integer success counts, and `exponential(lambda, shape)` returns
non-negative samples for a positive finite rate. All distributions advance
only their owning generator.
These methods return errors for invalid distribution parameters. Streams are
reproducible with the same V/VTL versions, seed, and sequence of calls;
cross-version compatibility is not guaranteed. See the complete data-pipeline
example in [`examples/random_generator/main.v`](../examples/random_generator/main.v).

VTL also provides global tensor constructors for uniform range, normal,
Bernoulli, binomial, geometric, and exponential distributions. See
[`rand.v`](../rand.v) for the complete API.

`random[T](minimum, maximum, shape, params)` creates uniform values in the
half-open range `[minimum, maximum)`. Integer tensors support `i8`, `i16`,
`i32`, `i64`, `int`, `u8`, `u16`, `u32`, and `u64`:

```v
import vtl

labels := vtl.random[u8](0, 10, [8], vtl.TensorData{})
for label in labels.to_array() {
	assert label >= 0 && label < 10
}
```
