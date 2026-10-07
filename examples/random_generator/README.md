# Independent random generators

Use `RandomGenerator` to keep seeded data-pipeline or experiment streams
independent from V's global random state. The example covers continuous and
discrete distributions, sampling, and model-training noise.

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/random_generator/main.v
```

The same seed and sequence of calls reproduce the same values for the same
VTL/V runtime version.
