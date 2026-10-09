# Multivariate normal samples

Draw correlated vectors from a positive-semidefinite covariance matrix using
VTL's seeded random generator. The output shape is `sample_shape + [n]`, where
`n` is the length of the mean vector.

Run from `~/.vmodules`:

```sh
v run ./vtl/examples/multivariate_normal/main.v
```

The example is CPU-only and uses the VSL symmetric eigensolver, so singular
covariance matrices are supported as well as positive-definite ones.
