module vtl

// QuantileMethod selects one of NumPy's thirteen sample-quantile estimators.
pub enum QuantileMethod {
	inverted_cdf
	averaged_inverted_cdf
	closest_observation
	interpolated_inverted_cdf
	hazen
	weibull
	linear
	median_unbiased
	normal_unbiased
	lower
	higher
	midpoint
	nearest
}
