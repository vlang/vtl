module stats

import vtl

// quantiles_axis_keepdims computes linear quantiles and retains the reduced axis.
pub fn quantiles_axis_keepdims[T](t &vtl.Tensor[T], quantiles []f64, axis int, keepdims bool) !&vtl.Tensor[f64] {
	return quantiles_axis_with_method_keepdims[T](t, quantiles, axis, .linear, keepdims)
}

// nanquantiles_axis_keepdims computes NaN-aware linear quantiles with retained dimensions.
pub fn nanquantiles_axis_keepdims[T](t &vtl.Tensor[T], quantiles []f64, axis int, keepdims bool) !&vtl.Tensor[f64] {
	return nanquantiles_axis_with_method_keepdims[T](t, quantiles, axis, .linear, keepdims)
}

// percentiles_axis_keepdims computes linear percentiles with retained dimensions.
pub fn percentiles_axis_keepdims[T](t &vtl.Tensor[T], percentiles []f64, axis int, keepdims bool) !&vtl.Tensor[f64] {
	return percentiles_axis_with_method_keepdims[T](t, percentiles, axis, .linear, keepdims)
}

// nanpercentiles_axis_keepdims computes NaN-aware linear percentiles with retained dimensions.
pub fn nanpercentiles_axis_keepdims[T](t &vtl.Tensor[T], percentiles []f64, axis int, keepdims bool) !&vtl.Tensor[f64] {
	return nanpercentiles_axis_with_method_keepdims[T](t, percentiles, axis, .linear, keepdims)
}
