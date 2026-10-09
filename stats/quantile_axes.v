module stats

import vtl

// quantile_axes reduces multiple axes using NumPy's default linear estimator.
pub fn quantile_axes[T](t &vtl.Tensor[T], q f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return quantile_axes_with_method[T](t, q, axes, .linear, keepdims)
}

// nanquantile_axes reduces multiple axes while ignoring NaNs, using linear estimation.
pub fn nanquantile_axes[T](t &vtl.Tensor[T], q f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return nanquantile_axes_with_method[T](t, q, axes, .linear, keepdims)
}

// quantiles_axes computes several linear quantiles per multi-axis slice.
pub fn quantiles_axes[T](t &vtl.Tensor[T], quantiles []f64, axes []int) !&vtl.Tensor[f64] {
	return quantiles_axes_with_method[T](t, quantiles, axes, .linear)
}

// quantiles_axes_keepdims retains reduced dimensions as length one.
pub fn quantiles_axes_keepdims[T](t &vtl.Tensor[T], quantiles []f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return quantiles_axes_with_method_keepdims[T](t, quantiles, axes, .linear, keepdims)
}

// nanquantiles_axes computes several NaN-aware linear quantiles per slice.
pub fn nanquantiles_axes[T](t &vtl.Tensor[T], quantiles []f64, axes []int) !&vtl.Tensor[f64] {
	return nanquantiles_axes_with_method[T](t, quantiles, axes, .linear)
}

// nanquantiles_axes_keepdims retains reduced dimensions as length one.
pub fn nanquantiles_axes_keepdims[T](t &vtl.Tensor[T], quantiles []f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return nanquantiles_axes_with_method_keepdims[T](t, quantiles, axes, .linear, keepdims)
}

// percentile_axes reduces multiple axes using a 0..100 percentile and linear estimation.
pub fn percentile_axes[T](t &vtl.Tensor[T], percentile f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return percentile_axes_with_method[T](t, percentile, axes, .linear, keepdims)
}

// nanpercentile_axes is the NaN-aware 0..100 percentile form.
pub fn nanpercentile_axes[T](t &vtl.Tensor[T], percentile f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return nanpercentile_axes_with_method[T](t, percentile, axes, .linear, keepdims)
}

// percentiles_axes computes multiple 0..100 percentiles using linear estimation.
pub fn percentiles_axes[T](t &vtl.Tensor[T], percentiles []f64, axes []int) !&vtl.Tensor[f64] {
	return percentiles_axes_with_method[T](t, percentiles, axes, .linear)
}

// percentiles_axes_keepdims retains reduced dimensions as length one.
pub fn percentiles_axes_keepdims[T](t &vtl.Tensor[T], percentiles []f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return percentiles_axes_with_method_keepdims[T](t, percentiles, axes, .linear, keepdims)
}

// nanpercentiles_axes computes multiple NaN-aware 0..100 percentiles.
pub fn nanpercentiles_axes[T](t &vtl.Tensor[T], percentiles []f64, axes []int) !&vtl.Tensor[f64] {
	return nanpercentiles_axes_with_method[T](t, percentiles, axes, .linear)
}

// nanpercentiles_axes_keepdims retains reduced dimensions as length one.
pub fn nanpercentiles_axes_keepdims[T](t &vtl.Tensor[T], percentiles []f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return nanpercentiles_axes_with_method_keepdims[T](t, percentiles, axes, .linear, keepdims)
}
