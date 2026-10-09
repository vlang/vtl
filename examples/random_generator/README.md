# Independent random generators

Use `RandomGenerator` to keep seeded data-pipeline or experiment streams
independent from V's global random state. The example covers continuous and
discrete distributions, seeded integer sampling, sampling, and model-training noise.
`permutation_axis` also shuffles complete slices along a chosen axis, including
negative axis indices, while preserving the tensor shape. `choice_axis` samples
complete slices with or without replacement and substitutes the sampled length
for the selected axis.

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/random_generator/main.v
```

The example includes Gumbel extreme-value, Laplace robust, Logistic, Pareto,
Rayleigh, and Triangular samples in addition to seeded normal, Bernoulli,
binomial, negative-binomial, hypergeometric, Poisson, Weibull, Gamma, Beta, and
Student distributions.
Hypergeometric sampling draws without replacement from good and bad population
counts. Location and scale
must be finite; scale may be zero to produce a constant tensor. The same seed
and sequence of calls reproduce the same values for the same VTL/V runtime
version.
