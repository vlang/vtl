# Fourier transforms

The optional `vtl.fft` module wraps VSL's PocketFFT backend for one-dimensional
real transforms. It accepts `f32` and `f64` input tensors and returns the
non-negative frequency bins as complex values.

```v
import vtl
import vtl.fft

signal := vtl.from_1d([0.5, 0.5, 1.0, 2.0])!
spectrum := fft.rfft[f64](signal)!
restored := fft.irfft(spectrum, signal.size)!
println(spectrum.shape) // [3], including DC and Nyquist bins
println(restored) // [0.5, 0.5, 1, 2]
```

`irfft` takes the original real length because an even and an odd signal can
have the same number of non-negative bins. It applies the NumPy-style inverse
normalization. The backend currently supports one-dimensional real transforms;
complex input and multidimensional transforms are not exposed yet.
