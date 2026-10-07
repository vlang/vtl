module vtl

import math

enum MaskedReduction {
	sum
	product
	minimum
	maximum
}

enum MaskedBinaryOperation {
	add
	subtract
	multiply
	divide
}

enum MaskedComparison {
	equal
	not_equal
	less
	less_equal
	greater
	greater_equal
}

// MaskedArray pairs tensor values with a boolean missing-data mask. A true
// mask entry marks the corresponding value as missing, following NumPy's
// numpy.ma convention.
pub struct MaskedArray[T] {
pub:
	values &Tensor[T]
	mask   &Tensor[bool]
}

// MaskedValue carries a reduction result and indicates whether it is masked.
pub struct MaskedValue[T] {
pub:
	value     T
	is_masked bool
}

struct MaskedMoment {
mut:
	count int
	mean  f64
	m2    f64
}

// masked_array creates a masked tensor. The mask broadcasts against values;
// the returned values and mask are views with the common broadcast shape.
pub fn masked_array[T](values &Tensor[T], mask &Tensor[bool]) !MaskedArray[T] {
	rank := math.max(values.rank(), mask.rank())
	mut values_shape := []int{len: rank, init: 1}
	mut mask_shape := []int{len: rank, init: 1}
	for i, size in values.shape {
		values_shape[rank - values.rank() + i] = size
	}
	for i, size in mask.shape {
		mask_shape[rank - mask.rank() + i] = size
	}
	if !broadcast_equal(values_shape, mask_shape) {
		return error('masked_array: values with shape ${values.shape} and mask with shape ${mask.shape} cannot broadcast')
	}
	mut shape := []int{len: rank}
	for i in 0 .. rank {
		if values_shape[i] == 1 {
			shape[i] = mask_shape[i]
		} else {
			shape[i] = values_shape[i]
		}
	}
	return MaskedArray[T]{
		values: values.broadcast_to[T](shape)!
		mask:   mask.broadcast_to[bool](shape)!
	}
}

// filled returns a row-major copy with masked entries replaced by fill_value.
pub fn (array &MaskedArray[T]) filled[T](fill_value T) &Tensor[T] {
	mut result := empty[T](array.values.shape, memory: .row_major)
	for i in 0 .. result.size {
		value := if array.mask.get_nth(i) { fill_value } else { array.values.get_nth(i) }
		result.set_nth(i, value)
	}
	return result
}

// compressed returns all unmasked values as a one-dimensional copy in
// row-major logical order.
pub fn (array &MaskedArray[T]) compressed[T]() !&Tensor[T] {
	mut visible := empty[bool](array.mask.shape, memory: .row_major)
	for i in 0 .. visible.size {
		visible.set_nth(i, !array.mask.get_nth(i))
	}
	return array.values.masked_select(visible)!
}

// mixed_index applies NumPy-style mixed basic and advanced indexing to both
// values and mask, preserving their alignment and view/copy behavior.
pub fn (array &MaskedArray[T]) mixed_index[T](indices []TensorIndex) !MaskedArray[T] {
	return MaskedArray[T]{
		values: mixed_index[T](array.values, indices)!
		mask:   mixed_index[bool](array.mask, indices)!
	}
}

// count returns the number of unmasked values.
pub fn (array &MaskedArray[T]) count[T]() int {
	mut valid := 0
	for i in 0 .. array.mask.size {
		if !array.mask.get_nth(i) {
			valid++
		}
	}
	return valid
}

// add adds two masked arrays with broadcasting and combines their masks.
pub fn (array &MaskedArray[T]) add[T](other &MaskedArray[T]) !MaskedArray[T] {
	return masked_binary[T](array, other, .add)
}

// subtract subtracts another masked array with broadcasting.
pub fn (array &MaskedArray[T]) subtract[T](other &MaskedArray[T]) !MaskedArray[T] {
	return masked_binary[T](array, other, .subtract)
}

// multiply multiplies two masked arrays with broadcasting.
pub fn (array &MaskedArray[T]) multiply[T](other &MaskedArray[T]) !MaskedArray[T] {
	return masked_binary[T](array, other, .multiply)
}

// divide divides by another masked array with broadcasting.
pub fn (array &MaskedArray[T]) divide[T](other &MaskedArray[T]) !MaskedArray[T] {
	return masked_binary[T](array, other, .divide)
}

// equal performs a broadcasted equality comparison and masks invalid pairs.
pub fn (array &MaskedArray[T]) equal[T](other &MaskedArray[T]) !MaskedArray[bool] {
	return masked_compare[T](array, other, .equal)
}

// not_equal performs a broadcasted inequality comparison and masks invalid pairs.
pub fn (array &MaskedArray[T]) not_equal[T](other &MaskedArray[T]) !MaskedArray[bool] {
	return masked_compare[T](array, other, .not_equal)
}

// less_than compares values with < and masks invalid pairs.
pub fn (array &MaskedArray[T]) less_than[T](other &MaskedArray[T]) !MaskedArray[bool] {
	return masked_compare[T](array, other, .less)
}

// less_equal compares values with <= and masks invalid pairs.
pub fn (array &MaskedArray[T]) less_equal[T](other &MaskedArray[T]) !MaskedArray[bool] {
	return masked_compare[T](array, other, .less_equal)
}

// greater_than compares values with > and masks invalid pairs.
pub fn (array &MaskedArray[T]) greater_than[T](other &MaskedArray[T]) !MaskedArray[bool] {
	return masked_compare[T](array, other, .greater)
}

// greater_equal compares values with >= and masks invalid pairs.
pub fn (array &MaskedArray[T]) greater_equal[T](other &MaskedArray[T]) !MaskedArray[bool] {
	return masked_compare[T](array, other, .greater_equal)
}

// add_scalar adds a valid scalar to every value and preserves the mask.
pub fn (array &MaskedArray[T]) add_scalar[T](scalar T) !MaskedArray[T] {
	return MaskedArray[T]{
		values: array.values.add_scalar[T](scalar)!
		mask:   array.mask
	}
}

// subtract_scalar subtracts a valid scalar from every value and preserves the mask.
pub fn (array &MaskedArray[T]) subtract_scalar[T](scalar T) !MaskedArray[T] {
	return MaskedArray[T]{
		values: array.values.subtract_scalar[T](scalar)!
		mask:   array.mask
	}
}

// multiply_scalar multiplies every value by a valid scalar and preserves the mask.
pub fn (array &MaskedArray[T]) multiply_scalar[T](scalar T) !MaskedArray[T] {
	return MaskedArray[T]{
		values: array.values.multiply_scalar[T](scalar)!
		mask:   array.mask
	}
}

// divide_scalar divides every value by a valid scalar and preserves the mask.
pub fn (array &MaskedArray[T]) divide_scalar[T](scalar T) !MaskedArray[T] {
	return MaskedArray[T]{
		values: array.values.divide_scalar[T](scalar)!
		mask:   array.mask
	}
}

fn masked_binary[T](array &MaskedArray[T], other &MaskedArray[T], operation MaskedBinaryOperation) !MaskedArray[T] {
	values := if operation == .add {
		array.values.add[T](other.values)!
	} else if operation == .subtract {
		array.values.subtract[T](other.values)!
	} else if operation == .multiply {
		array.values.multiply[T](other.values)!
	} else {
		array.values.divide[T](other.values)!
	}
	mask := array.mask.logical_or[bool](other.mask)!
	return masked_array[T](values, mask)
}

fn masked_compare[T](array &MaskedArray[T], other &MaskedArray[T], operation MaskedComparison) !MaskedArray[bool] {
	mut iterators, shape := array.values.iterators[T]([other.values])!
	mut values := empty[bool](shape)
	for {
		pair, index := iterators.next() or { break }
		left := pair[0]
		right := pair[1]
		result := if operation == .equal {
			left == right
		} else if operation == .not_equal {
			left != right
		} else if operation == .less {
			left < right
		} else if operation == .less_equal {
			left <= right
		} else if operation == .greater {
			left > right
		} else {
			left >= right
		}
		values.set(index, result)
	}
	mask := array.mask.logical_or[bool](other.mask)!
	return masked_array[bool](values, mask)
}

// sum reduces all unmasked values. An entirely masked array returns a masked
// additive identity with value zero.
pub fn (array &MaskedArray[T]) sum[T]() MaskedValue[T] {
	mut total := cast[T](0)
	mut count := 0
	for i in 0 .. array.values.size {
		if !array.mask.get_nth(i) {
			total += array.values.get_nth(i)
			count++
		}
	}
	return MaskedValue[T]{
		value:     total
		is_masked: count == 0
	}
}

// prod reduces all unmasked values. An entirely masked array returns a masked
// multiplicative identity with value one.
pub fn (array &MaskedArray[T]) prod[T]() MaskedValue[T] {
	mut total := cast[T](1)
	mut count := 0
	for i in 0 .. array.values.size {
		if !array.mask.get_nth(i) {
			total *= array.values.get_nth(i)
			count++
		}
	}
	return MaskedValue[T]{
		value:     total
		is_masked: count == 0
	}
}

// min returns the smallest unmasked value. An entirely masked array returns a
// masked result whose value is zero and must be ignored.
pub fn (array &MaskedArray[T]) min[T]() MaskedValue[T] {
	return masked_extreme[T](array, false)
}

// max returns the largest unmasked value. An entirely masked array returns a
// masked result whose value is zero and must be ignored.
pub fn (array &MaskedArray[T]) max[T]() MaskedValue[T] {
	return masked_extreme[T](array, true)
}

fn masked_extreme[T](array &MaskedArray[T], maximum bool) MaskedValue[T] {
	mut result := cast[T](0)
	mut count := 0
	for i in 0 .. array.values.size {
		if array.mask.get_nth(i) {
			continue
		}
		value := array.values.get_nth(i)
		if count == 0 || (maximum && value > result) || (!maximum && value < result) {
			result = value
		}
		count++
	}
	return MaskedValue[T]{
		value:     result
		is_masked: count == 0
	}
}

// mean reduces all unmasked values. An entirely masked array returns a masked
// NaN result.
pub fn (array &MaskedArray[T]) mean[T]() MaskedValue[f64] {
	mut total := 0.0
	mut count := 0
	for i in 0 .. array.values.size {
		if !array.mask.get_nth(i) {
			total += td(array.values.get_nth(i)).f64()
			count++
		}
	}
	return MaskedValue[f64]{
		value:     if count == 0 { math.nan() } else { total / f64(count) }
		is_masked: count == 0
	}
}

// variance computes population or sample variance over unmasked values. ddof
// is subtracted from the valid count; insufficient data returns a masked NaN.
pub fn (array &MaskedArray[T]) variance[T](ddof int) !MaskedValue[f64] {
	validate_masked_ddof(ddof)!
	mut moment := MaskedMoment{}
	for i in 0 .. array.values.size {
		if !array.mask.get_nth(i) {
			moment.add(td(array.values.get_nth(i)).f64())
		}
	}
	if moment.count <= ddof {
		return MaskedValue[f64]{
			value:     math.nan()
			is_masked: true
		}
	}
	return MaskedValue[f64]{
		value:     moment.m2 / f64(moment.count - ddof)
		is_masked: false
	}
}

// std computes population or sample standard deviation over unmasked values.
pub fn (array &MaskedArray[T]) std[T](ddof int) !MaskedValue[f64] {
	variance := array.variance[T](ddof)!
	return MaskedValue[f64]{
		value:     math.sqrt(variance.value)
		is_masked: variance.is_masked
	}
}

// sum_along_axis reduces unmasked values along one axis. Output mask entries
// are true for slices containing no unmasked values.
pub fn (array &MaskedArray[T]) sum_along_axis[T](axis int, keepdims bool) !MaskedArray[T] {
	return array.sum_along_axes[T]([axis], keepdims)
}

// sum_along_axes reduces unmasked values over several axes. Negative axes are
// supported; duplicate axes are rejected. An empty axes list preserves data.
pub fn (array &MaskedArray[T]) sum_along_axes[T](axes []int, keepdims bool) !MaskedArray[T] {
	if axes.len == 0 {
		return *array
	}
	return masked_reduce_axes[T](array, axes, keepdims, .sum)
}

// prod_along_axis multiplies unmasked values along one axis.
pub fn (array &MaskedArray[T]) prod_along_axis[T](axis int, keepdims bool) !MaskedArray[T] {
	return array.prod_along_axes[T]([axis], keepdims)
}

// prod_along_axes multiplies unmasked values over several axes. Negative axes
// are supported; duplicate axes are rejected. An empty axes list preserves data.
pub fn (array &MaskedArray[T]) prod_along_axes[T](axes []int, keepdims bool) !MaskedArray[T] {
	if axes.len == 0 {
		return *array
	}
	return masked_reduce_axes[T](array, axes, keepdims, .product)
}

// min_along_axis finds the smallest unmasked value in each axis slice.
pub fn (array &MaskedArray[T]) min_along_axis[T](axis int, keepdims bool) !MaskedArray[T] {
	return array.min_along_axes[T]([axis], keepdims)
}

// min_along_axes finds minima across several axes. Empty slices are masked.
pub fn (array &MaskedArray[T]) min_along_axes[T](axes []int, keepdims bool) !MaskedArray[T] {
	if axes.len == 0 {
		return *array
	}
	return masked_reduce_axes[T](array, axes, keepdims, .minimum)
}

// max_along_axis finds the largest unmasked value in each axis slice.
pub fn (array &MaskedArray[T]) max_along_axis[T](axis int, keepdims bool) !MaskedArray[T] {
	return array.max_along_axes[T]([axis], keepdims)
}

// max_along_axes finds maxima across several axes. Empty slices are masked.
pub fn (array &MaskedArray[T]) max_along_axes[T](axes []int, keepdims bool) !MaskedArray[T] {
	if axes.len == 0 {
		return *array
	}
	return masked_reduce_axes[T](array, axes, keepdims, .maximum)
}

// variance_along_axis computes variance along one axis, marking slices with
// insufficient valid values as masked NaN results.
pub fn (array &MaskedArray[T]) variance_along_axis[T](axis int, ddof int, keepdims bool) !MaskedArray[f64] {
	return array.variance_along_axes[T]([axis], ddof, keepdims)
}

// variance_along_axes computes population or sample variance over several
// axes. Empty axes compute elementwise zero variance for valid values.
pub fn (array &MaskedArray[T]) variance_along_axes[T](axes []int, ddof int, keepdims bool) !MaskedArray[f64] {
	return masked_moment_reduction[T](array, axes, ddof, keepdims, false)
}

// std_along_axis computes standard deviation along one axis.
pub fn (array &MaskedArray[T]) std_along_axis[T](axis int, ddof int, keepdims bool) !MaskedArray[f64] {
	return array.std_along_axes[T]([axis], ddof, keepdims)
}

// std_along_axes computes standard deviation over several axes.
pub fn (array &MaskedArray[T]) std_along_axes[T](axes []int, ddof int, keepdims bool) !MaskedArray[f64] {
	return masked_moment_reduction[T](array, axes, ddof, keepdims, true)
}

fn masked_moment_reduction[T](array &MaskedArray[T], axes []int, ddof int, keepdims bool, standard_deviation bool) !MaskedArray[f64] {
	validate_masked_ddof(ddof)!
	mut reduced := []bool{len: array.values.rank()}
	if axes.len > 0 {
		reduced = normalize_masked_axes(axes, array.values.rank())!
	}
	output_shape := masked_reduction_shape(array.values.shape, reduced, keepdims)
	mut values := zeros[f64](output_shape, TensorData{})
	mut output_mask := ones[bool](output_shape, TensorData{})
	mut moments := []MaskedMoment{len: values.size}
	mut input_index := []int{len: array.values.rank()}
	for flat_index in 0 .. array.values.size {
		decode_flat_coordinate(flat_index, array.values.shape, mut input_index)
		if array.mask.get_nth(flat_index) {
			continue
		}
		output_flat_index := masked_axes_output_index(input_index, array.values.shape, reduced)
		moments[output_flat_index].add(td(array.values.get_nth(flat_index)).f64())
	}
	for i in 0 .. values.size {
		moment := moments[i]
		if moment.count > ddof {
			variance := moment.m2 / f64(moment.count - ddof)
			values.set_nth(i, if standard_deviation { math.sqrt(variance) } else { variance })
			output_mask.set_nth(i, false)
		} else {
			values.set_nth(i, math.nan())
		}
	}
	return MaskedArray[f64]{
		values: values
		mask:   output_mask
	}
}

fn validate_masked_ddof(ddof int) ! {
	if ddof < 0 {
		return error('ddof must be non-negative')
	}
}

fn (mut moment MaskedMoment) add(value f64) {
	moment.count++
	delta := value - moment.mean
	moment.mean += delta / f64(moment.count)
	moment.m2 += delta * (value - moment.mean)
}

fn masked_reduce_axes[T](array &MaskedArray[T], axes []int, keepdims bool, operation MaskedReduction) !MaskedArray[T] {
	reduced := normalize_masked_axes(axes, array.values.rank())!
	output_shape := masked_reduction_shape(array.values.shape, reduced, keepdims)
	mut values := if operation == .product {
		ones[T](output_shape, TensorData{})
	} else {
		zeros[T](output_shape, TensorData{})
	}
	mut output_mask := ones[bool](output_shape, TensorData{})
	mut counts := []int{len: values.size}
	mut input_index := []int{len: array.values.rank()}
	for flat_index in 0 .. array.values.size {
		decode_flat_coordinate(flat_index, array.values.shape, mut input_index)
		if array.mask.get_nth(flat_index) {
			continue
		}
		output_flat_index := masked_axes_output_index(input_index, array.values.shape, reduced)
		value := array.values.get_nth(flat_index)
		if operation == .sum {
			values.set_nth(output_flat_index, values.get_nth(output_flat_index) + value)
		} else if operation == .product {
			values.set_nth(output_flat_index, values.get_nth(output_flat_index) * value)
		} else if operation == .minimum {
			if counts[output_flat_index] == 0 || value < values.get_nth(output_flat_index) {
				values.set_nth(output_flat_index, value)
			}
		} else {
			if counts[output_flat_index] == 0 || value > values.get_nth(output_flat_index) {
				values.set_nth(output_flat_index, value)
			}
		}
		counts[output_flat_index]++
	}
	for i in 0 .. values.size {
		output_mask.set_nth(i, counts[i] == 0)
	}
	return MaskedArray[T]{
		values: values
		mask:   output_mask
	}
}

// mean_along_axis computes the mean of unmasked values per axis slice. Slices
// with no unmasked values carry a masked NaN result.
pub fn (array &MaskedArray[T]) mean_along_axis[T](axis int, keepdims bool) !MaskedArray[f64] {
	return array.mean_along_axes[T]([axis], keepdims)
}

// mean_along_axes computes means over several axes. Negative axes are
// supported; duplicate axes are rejected. An empty axes list converts each
// unmasked value to f64 and preserves the mask.
pub fn (array &MaskedArray[T]) mean_along_axes[T](axes []int, keepdims bool) !MaskedArray[f64] {
	if axes.len == 0 {
		mut values := empty[f64](array.values.shape, memory: .row_major)
		for i in 0 .. array.values.size {
			values.set_nth(i, td(array.values.get_nth(i)).f64())
		}
		return MaskedArray[f64]{
			values: values
			mask:   array.mask
		}
	}
	reduced := normalize_masked_axes(axes, array.values.rank())!
	output_shape := masked_reduction_shape(array.values.shape, reduced, keepdims)
	mut values := zeros[f64](output_shape, TensorData{})
	mut output_mask := ones[bool](output_shape, TensorData{})
	mut totals := []f64{len: values.size}
	mut counts := []int{len: values.size}
	mut input_index := []int{len: array.values.rank()}
	for flat_index in 0 .. array.values.size {
		decode_flat_coordinate(flat_index, array.values.shape, mut input_index)
		if array.mask.get_nth(flat_index) {
			continue
		}
		output_flat_index := masked_axes_output_index(input_index, array.values.shape, reduced)
		totals[output_flat_index] += td(array.values.get_nth(flat_index)).f64()
		counts[output_flat_index]++
	}
	for i in 0 .. values.size {
		if counts[i] > 0 {
			values.set_nth(i, totals[i] / f64(counts[i]))
			output_mask.set_nth(i, false)
		} else {
			values.set_nth(i, math.nan())
		}
	}
	return MaskedArray[f64]{
		values: values
		mask:   output_mask
	}
}

fn normalize_masked_axes(axes []int, rank int) ![]bool {
	if rank == 0 {
		return error('masked axis reduction requires at least one dimension')
	}
	mut reduced := []bool{len: rank}
	for axis in axes {
		axis_index := if axis < 0 { axis + rank } else { axis }
		if axis_index < 0 || axis_index >= rank {
			return error('axis ${axis} out of bounds for rank ${rank}')
		}
		if reduced[axis_index] {
			return error('axis ${axis} appears more than once')
		}
		reduced[axis_index] = true
	}
	return reduced
}

fn masked_reduction_shape(input_shape []int, reduced []bool, keepdims bool) []int {
	mut output_shape := []int{cap: input_shape.len}
	for dimension, size in input_shape {
		if reduced[dimension] {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << size
		}
	}
	return output_shape
}

fn masked_axes_output_index(input_index []int, input_shape []int, reduced []bool) int {
	mut output_flat_index := 0
	for dimension, coordinate in input_index {
		if !reduced[dimension] {
			output_flat_index = output_flat_index * input_shape[dimension] + coordinate
		}
	}
	return output_flat_index
}
