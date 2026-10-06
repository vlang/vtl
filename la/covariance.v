module la

import math
import vtl

// covariance_matrix returns a variables-by-variables covariance matrix.
// Input is interpreted like NumPy's cov: rows are variables when rowvar is
// true, and columns are variables when rowvar is false. ddof is the delta
// degrees of freedom: use 0 for population covariance and 1 for sample.
pub fn covariance_matrix[T](data &vtl.Tensor[T], rowvar bool, ddof int) !&vtl.Tensor[f64] {
	if data.rank() != 2 {
		return error('covariance_matrix expects a two-dimensional tensor')
	}
	variables := if rowvar { data.shape[0] } else { data.shape[1] }
	observations := if rowvar { data.shape[1] } else { data.shape[0] }
	if variables == 0 || observations == 0 {
		return error('covariance_matrix requires variables and at least one observation')
	}
	mut means := []f64{len: variables}
	for variable in 0 .. variables {
		mut sum := 0.0
		for observation in 0 .. observations {
			sum += covariance_value[T](data, rowvar, variable, observation)
		}
		means[variable] = sum / f64(observations)
	}
	mut result := []f64{len: variables * variables}
	divisor := f64(observations - ddof)
	for left in 0 .. variables {
		for right in left .. variables {
			mut sum := 0.0
			for observation in 0 .. observations {
				x := covariance_value[T](data, rowvar, left, observation) - means[left]
				y := covariance_value[T](data, rowvar, right, observation) - means[right]
				sum += x * y
			}
			value := if divisor == 0 { math.nan() } else { sum / divisor }
			result[left * variables + right] = value
			result[right * variables + left] = value
		}
	}
	return vtl.from_array[f64](result, [variables, variables])
}

// correlation_matrix returns the Pearson correlation matrix. As with
// covariance_matrix, rowvar selects whether observations occupy rows or
// columns. Constant variables produce NaN correlations, matching NumPy.
pub fn correlation_matrix[T](data &vtl.Tensor[T], rowvar bool) !&vtl.Tensor[f64] {
	covariance := covariance_matrix[T](data, rowvar, 1)!
	variables := covariance.shape[0]
	mut result := []f64{len: variables * variables}
	for left in 0 .. variables {
		left_std := math.sqrt(covariance.get([left, left]))
		for right in 0 .. variables {
			right_std := math.sqrt(covariance.get([right, right]))
			if left_std == 0 || right_std == 0 {
				result[left * variables + right] = math.nan()
			} else {
				result[left * variables + right] = covariance.get([left, right]) / (left_std * right_std)
			}
		}
	}
	return vtl.from_array[f64](result, [variables, variables])
}

fn covariance_value[T](data &vtl.Tensor[T], rowvar bool, variable int, observation int) f64 {
	value := if rowvar {
		data.get([variable, observation])
	} else {
		data.get([observation, variable])
	}
	return f64(value)
}
