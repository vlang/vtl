# Fourier transforms

The optional `vtl.fft` module wraps VSL's PocketFFT backend for one-dimensional
real and complex transforms. Real input accepts `f32` and `f64` tensors and
returns the non-negative frequency bins as complex values. Full complex
transforms use `math.complex.Complex` values.

```v
import vtl
import vtl.fft
import math.complex

signal := vtl.from_1d([0.5, 0.5, 1.0, 2.0])!
spectrum := fft.rfft[f64](signal)!
restored := fft.irfft(spectrum, signal.size)!
println(spectrum.shape) // [3], including DC and Nyquist bins
println(restored) // [0.5, 0.5, 1, 2]

mut plan := fft.create_rfft_plan[f64](signal.size)!
defer {
	plan.destroy()
}
repeated_spectrum := plan.forward(signal)!

complex_signal := vtl.from_1d[complex.Complex]([
	complex.complex(1, 0),
	complex.complex(0, 0),
])!
complex_spectrum := fft.fft(complex_signal)!
complex_restored := fft.ifft(complex_spectrum)!

image := vtl.from_array[complex.Complex]([
	complex.complex(1, 0),
	complex.complex(0, 0),
	complex.complex(0, 0),
	complex.complex(0, 0),
], [2, 2])!
image_spectrum := fft.fft2(image)!
image_restored := fft.ifft2(image_spectrum)!

real_image := vtl.from_array([f64(1), 0, 0, 0], [2, 2])!
real_spectrum := fft.rfft2[f64](real_image)!
real_restored := fft.irfft2(real_spectrum, real_image.shape)!
```

`irfft` takes the original real length because an even and an odd signal can
have the same number of non-negative bins. It applies the NumPy-style inverse
normalization. `fft2` and `ifft2` operate on two-dimensional complex tensors;
`fftn` and `ifftn` apply complex transforms across every tensor axis. `rfft2`
and `irfft2` cover two-dimensional real transforms, while `rfftn` and `irfftn`
support real transforms over all axes. The original shape is required by the
inverse real operations to disambiguate odd and even final dimensions.