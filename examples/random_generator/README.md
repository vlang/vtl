# Independent random generators

Use `RandomGenerator` to keep seeded data-pipeline or experiment streams
independent from V's global random state. The example covers continuous and
discrete distributions, sampling, and model-training noise.
`permutation_axis` also shuffles complete slices along a chosen axis, including
negative axis indices, while preserving the tensor shape. `choice_axis` samples
complete slices with or without replacement and substitutes the sampled length
for the selected axis.

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/random_generator/main.v
```

The same seed and sequence of calls reproduce the same values for the same
VTL/V runtime version.
