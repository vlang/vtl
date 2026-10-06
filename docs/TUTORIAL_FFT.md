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
frequencies := fft.fftfreq(complex_signal.size, 1.0)!
centered_spectrum := fft.fftshift(complex_spectrum)!

image := vtl.from_array[complex.Complex]([
	complex.complex(1, 0),
	complex.complex(0, 0),
	complex.complex(0, 0),
	complex.complex(0, 0),
], [2, 2])!
image_spectrum := fft.fft2(image)!
image_restored := fft.ifft2(image_spectrum)!
column_spectrum := fft.fft_axis(image, -1)! // transform the final axis only
column_restored := fft.ifft_axis(column_spectrum, 1)!

real_image := vtl.from_array([f64(1), 0, 0, 0], [2, 2])!
real_spectrum := fft.rfft2[f64](real_image)!
real_restored := fft.irfft2(real_spectrum, real_image.shape)!
row_spectrum := fft.rfft_axis(real_image, 1)!
row_restored := fft.irfft_axis(row_spectrum, 1, real_image.shape[1])!
```

`RealFftPlan.forward` reuses a per-plan work buffer and therefore takes a
mutable plan. Do not call `forward` concurrently on the same plan; create one
plan per concurrent worker.

`irfft` takes the original real length because an even and an odd signal can
have the same number of non-negative bins. It applies the NumPy-style inverse
normalization. `fft2` and `ifft2` operate on two-dimensional complex tensors;
`fftn` and `ifftn` apply complex transforms across every tensor axis. `rfft2`
and `irfft2` cover two-dimensional real transforms, while `rfftn` and `irfftn`
support real transforms over all axes. The original shape is required by the
inverse real operations to disambiguate odd and even final dimensions.

`fftfreq(n, d)` returns the full transform's frequency bins in FFT order;
`rfftfreq(n, d)` returns the non-negative bins for a real transform. `fftshift`
moves the zero-frequency component to the center of every axis, and
`ifftshift` reverses it. The `_axis` variants shift only one selected axis and
accept negative axis indices. Frequency helpers require a positive transform
length and finite, non-zero sample spacing.

`fft_axis` and `ifft_axis` transform just one axis of a complex tensor and
preserve its shape. Negative axis indices count backward from the final axis.
`rfft_axis` and `irfft_axis` do the same for a real tensor and its compact
complex spectrum. Pass the original real axis length to `irfft_axis` so odd
and even inputs can be distinguished.
