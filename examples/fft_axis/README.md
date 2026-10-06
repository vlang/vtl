# FFT along a selected axis

This example transforms a two-dimensional complex tensor along its final axis,
then applies the normalized inverse along that axis. It also computes a compact
real-input spectrum along one axis and reconstructs it using the original axis
length. Negative axis indices are supported for both operations.

Run it from the V module root:

```sh
v run ./vtl/examples/fft_axis/main.v
```
