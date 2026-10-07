module vtl

import math
import rand
import rand.config

// RandomGenerator owns an independent pseudorandom stream for reproducible
// experiments that must not modify or depend on V's global random state.
pub struct RandomGenerator {
mut:
	rng &rand.PRNG
}

// new_random_generator creates an independent generator initialized from seed.
pub fn new_random_generator(seed int) RandomGenerator {
	mut rng := rand.new_default(config.PRNGConfigStruct{
		seed_: [u32(seed), 0]
	})
	return RandomGenerator{
		rng: rng
	}
}

// free releases the generator's underlying pseudorandom engine.
pub fn (mut generator RandomGenerator) free() {
	generator.rng.free()
}

// uniform returns f64 values in the [minimum, maximum) range.
pub fn (mut generator RandomGenerator) uniform(minimum f64, maximum f64, shape []int) !&Tensor[f64] {
	if maximum < minimum {
		return error('uniform: maximum must be greater than or equal to minimum')
	}
	mut result := zeros[f64](shape, TensorData{})
	for i in 0 .. result.size {
		result.set_nth(i, generator.rng.f64_in_range(minimum, maximum)!)
	}
	return result
}

// normal returns f64 values from a normal distribution with the given parameters.
pub fn (mut generator RandomGenerator) normal(shape []int, params NormalTensorData) !&Tensor[f64] {
	if params.sigma <= 0 {
		return error('normal: sigma must be positive')
	}
	mut result := zeros[f64](shape, TensorData{})
	for i in 0 .. result.size {
		result.set_nth(i, generator.rng.normal(config.NormalConfigStruct{
			mu:    params.mu
			sigma: params.sigma
		})!)
	}
	return result
}

// lognormal returns samples whose natural logarithm follows a normal
// distribution with the given mean and standard deviation.
pub fn (mut generator RandomGenerator) lognormal(mean f64, sigma f64, shape []int) !&Tensor[f64] {
	if math.is_nan(mean) || math.is_inf(mean, 0) || sigma < 0 || math.is_nan(sigma)
		|| math.is_inf(sigma, 0) {
		return error('lognormal: mean must be finite and sigma must be finite and non-negative')
	}
	mut result := zeros[f64](shape, TensorData{})
	for i in 0 .. result.size {
		log_value := if sigma == 0 {
			mean
		} else {
			generator.rng.normal(config.NormalConfigStruct{
				mu:    mean
				sigma: sigma
			})!
		}
		result.set_nth(i, math.exp(log_value))
	}
	return result
}

// bernoulli returns bool values sampled with the given probability of true.
pub fn (mut generator RandomGenerator) bernoulli(probability f64, shape []int) !&Tensor[bool] {
	if probability < 0 || probability > 1 {
		return error('bernoulli: probability must be in [0, 1]')
	}
	mut result := zeros[bool](shape, TensorData{})
	for i in 0 .. result.size {
		result.set_nth(i, generator.rng.bernoulli(probability)!)
	}
	return result
}

// binomial returns the number of successful trials in each sample using this
// generator's independent stream.
pub fn (mut generator RandomGenerator) binomial(trials int, probability f64, shape []int) !&Tensor[int] {
	if trials < 0 {
		return error('binomial: trials must be non-negative')
	}
	if probability < 0 || probability > 1 || math.is_nan(probability) || math.is_inf(probability, 0) {
		return error('binomial: probability must be finite and in [0, 1]')
	}
	mut result := zeros[int](shape, TensorData{})
	for i in 0 .. result.size {
		result.set_nth(i, generator.rng.binomial(trials, probability)!)
	}
	return result
}

// multinomial returns category counts for repeated samples from one
// categorical distribution. The final output axis contains the categories.
pub fn (mut generator RandomGenerator) multinomial(trials int, probabilities &Tensor[f64], sample_shape []int) !&Tensor[int] {
	if trials < 0 {
		return error('multinomial: trials must be non-negative')
	}
	if probabilities.size == 0 {
		return error('multinomial: probabilities must not be empty')
	}
	mut values := probabilities.to_array()
	mut total_probability := 0.0
	for probability in values {
		if probability < 0 || math.is_nan(probability) || math.is_inf(probability, 0) {
			return error('multinomial: probabilities must be finite and non-negative')
		}
		total_probability += probability
	}
	if total_probability <= 0 || math.is_inf(total_probability, 0) {
		return error('multinomial: probabilities must have a finite positive sum')
	}
	if math.abs(total_probability - 1.0) > 1e-12 {
		return error('multinomial: probabilities must sum to one')
	}
	for i in 0 .. values.len {
		values[i] /= total_probability
	}
	category_count := values.len
	mut output_shape := sample_shape.clone()
	output_shape << category_count
	mut counts := []int{len: size_from_shape(output_shape)}
	sample_count := size_from_shape(sample_shape)
	for sample in 0 .. sample_count {
		mut remaining_trials := trials
		mut remaining_probability := 1.0
		for category in 0 .. category_count - 1 {
			probability := if remaining_probability <= 0 {
				0.0
			} else {
				values[category] / remaining_probability
			}
			count := generator.rng.binomial(remaining_trials, math.min(probability, 1.0))!
			counts[sample * category_count + category] = count
			remaining_trials -= count
			remaining_probability -= values[category]
		}
		counts[sample * category_count + category_count - 1] = remaining_trials
	}
	return from_array[int](counts, output_shape)
}

// exponential returns f64 samples from an exponential distribution with the
// given positive finite rate using this generator's independent stream.
pub fn (mut generator RandomGenerator) exponential(lambda f64, shape []int) !&Tensor[f64] {
	if lambda <= 0 || math.is_nan(lambda) || math.is_inf(lambda, 0) {
		return error('exponential: lambda must be finite and positive')
	}
	mut result := zeros[f64](shape, TensorData{})
	for i in 0 .. result.size {
		result.set_nth(i, generator.rng.exponential(lambda))
	}
	return result
}

// geometric returns the number of Bernoulli trials needed for the first
// success, independently sampled from this generator. Results start at 1.
pub fn (mut generator RandomGenerator) geometric(probability f64, shape []int) !&Tensor[int] {
	validate_geometric_probability(probability)!
	mut result := zeros[int](shape, TensorData{})
	for i in 0 .. result.size {
		u := generator.rng.f64_in_range(0.0, 1.0)!
		result.set_nth(i, geometric_sample(probability, u))
	}
	return result
}

// poisson returns integer event counts with the given finite, non-negative
// expected rate using this generator's independent stream.
pub fn (mut generator RandomGenerator) poisson(lambda f64, shape []int) !&Tensor[int] {
	validate_poisson_rate(lambda)!
	mut rng := generator.rng
	mut values := []int{len: size_from_shape(shape)}
	for i in 0 .. values.len {
		values[i] = sample_poisson(lambda, mut rng)!
	}
	return from_array[int](values, shape)
}

fn sample_poisson(lambda f64, mut rng &rand.PRNG) !int {
	if lambda == 0 {
		return 0
	}
	// Knuth's product method is simple and fast when the expected count is small.
	if lambda < 10 {
		limit := math.exp(-lambda)
		mut product := 1.0
		mut count := 0
		for product > limit {
			count++
			product *= rng.f64_in_range(0.0, 1.0)!
		}
		return count - 1
	}
	// Hörmann's PTRS transformed rejection method has constant expected work
	// for large rates, unlike the product method whose cost grows with lambda.
	ptrs := poisson_ptrs_config(lambda)
	for _ in 0 .. 10000 {
		u := rng.f64_in_range(0.0, 1.0)!
		v := rng.f64_in_range(0.0, 1.0)!
		candidate, accepted := sample_poisson_ptrs_candidate(ptrs, u, v)
		if accepted {
			return candidate
		}
	}
	return error('poisson: rejection sampler did not converge')
}

// weibull returns samples from the unit-scale Weibull distribution. The
// positive shape parameter controls the distribution's tail and hazard rate.
pub fn (mut generator RandomGenerator) weibull(shape_parameter f64, shape []int) !&Tensor[f64] {
	if shape_parameter <= 0 || math.is_nan(shape_parameter) || math.is_inf(shape_parameter, 0) {
		return error('weibull: shape parameter must be finite and positive')
	}
	mut values := []f64{len: size_from_shape(shape)}
	for i in 0 .. values.len {
		u := generator.rng.f64_in_range(0.0, 1.0)!
		values[i] = math.pow(-math.log(1.0 - u), 1.0 / shape_parameter)
	}
	return from_array[f64](values, shape)
}

// chi_square returns samples from a chi-square distribution with the given
// positive degrees of freedom.
pub fn (mut generator RandomGenerator) chi_square(degrees_of_freedom f64, shape []int) !&Tensor[f64] {
	validate_degrees_of_freedom(degrees_of_freedom, 'chi_square')!
	mut rng := generator.rng
	mut values := []f64{len: size_from_shape(shape)}
	for i in 0 .. values.len {
		values[i] = sample_gamma(degrees_of_freedom / 2, 2, mut rng)!
	}
	return from_array[f64](values, shape)
}

// student_t returns samples from the standard Student's t distribution.
pub fn (mut generator RandomGenerator) student_t(degrees_of_freedom f64, shape []int) !&Tensor[f64] {
	validate_degrees_of_freedom(degrees_of_freedom, 'student_t')!
	mut rng := generator.rng
	mut values := []f64{len: size_from_shape(shape)}
	for i in 0 .. values.len {
		normal_sample := rng.normal(config.NormalConfigStruct{})!
		chi_square_sample := sample_gamma(degrees_of_freedom / 2, 2, mut rng)!
		if chi_square_sample <= 0 {
			return error('student_t: sampled chi-square value underflowed to zero')
		}
		values[i] = normal_sample / math.sqrt(chi_square_sample / degrees_of_freedom)
	}
	return from_array[f64](values, shape)
}

// f_distribution returns samples from an F distribution with the specified
// numerator and denominator degrees of freedom.
pub fn (mut generator RandomGenerator) f_distribution(numerator_df f64, denominator_df f64, shape []int) !&Tensor[f64] {
	validate_degrees_of_freedom(numerator_df, 'f_distribution')!
	validate_degrees_of_freedom(denominator_df, 'f_distribution')!
	mut rng := generator.rng
	mut values := []f64{len: size_from_shape(shape)}
	for i in 0 .. values.len {
		numerator := sample_gamma(numerator_df / 2, 2, mut rng)!
		denominator := sample_gamma(denominator_df / 2, 2, mut rng)!
		if denominator <= 0 {
			return error('f_distribution: sampled denominator underflowed to zero')
		}
		values[i] = (numerator / numerator_df) / (denominator / denominator_df)
	}
	return from_array[f64](values, shape)
}

fn validate_degrees_of_freedom(degrees_of_freedom f64, distribution string) ! {
	if degrees_of_freedom <= 0 || math.is_nan(degrees_of_freedom)
		|| math.is_inf(degrees_of_freedom, 0) {
		return error('${distribution}: degrees of freedom must be finite and positive')
	}
}

// choice samples values from a tensor's flattened logical order. Without
// replacement, selected positions are unique; the same value may still occur
// more than once in the population.
pub fn (mut generator RandomGenerator) choice[T](population &Tensor[T], size int, replace bool) !&Tensor[T] {
	if population.size == 0 {
		return error('choice: population must not be empty')
	}
	if size < 0 {
		return error('choice: sample size must be non-negative')
	}
	if !replace && size > population.size {
		return error('choice: cannot sample more positions than the population without replacement')
	}
	mut selected := []T{len: size}
	if replace {
		for i in 0 .. size {
			index := int(generator.rng.f64_in_range(0.0, f64(population.size))!)
			selected[i] = population.get_nth(index)
		}
	} else {
		mut positions := []int{len: population.size}
		for index in 0 .. population.size {
			positions[index] = index
		}
		for i in 0 .. size {
			offset := int(generator.rng.f64_in_range(0.0, f64(population.size - i))!)
			selected_position := i + offset
			positions[i], positions[selected_position] = positions[selected_position], positions[i]
			selected[i] = population.get_nth(positions[i])
		}
	}
	return from_1d[T](selected)
}

// choice_weighted samples values from a tensor's flattened logical order in
// proportion to non-negative weights. Without replacement, each selected
// position is removed from the remaining weight distribution.
pub fn (mut generator RandomGenerator) choice_weighted[T](population &Tensor[T], weights &Tensor[f64], size int, replace bool) !&Tensor[T] {
	if population.size == 0 {
		return error('choice_weighted: population must not be empty')
	}
	if weights.size != population.size {
		return error('choice_weighted: weights must match the flattened population size')
	}
	if size < 0 {
		return error('choice_weighted: sample size must be non-negative')
	}
	if !replace && size > population.size {
		return error('choice_weighted: cannot sample more positions than the population without replacement')
	}
	mut remaining := weights.to_array()
	mut total_weight := 0.0
	mut positive_weights := 0
	for weight in remaining {
		if weight < 0 || math.is_nan(weight) || math.is_inf(weight, 0) {
			return error('choice_weighted: weights must be finite and non-negative')
		}
		total_weight += weight
		if weight > 0 {
			positive_weights++
		}
	}
	if total_weight <= 0 || math.is_inf(total_weight, 0) {
		return error('choice_weighted: weights must have a finite positive sum')
	}
	if !replace && size > positive_weights {
		return error('choice_weighted: not enough positive weights for sampling without replacement')
	}
	mut selected := []T{len: size}
	for draw in 0 .. size {
		threshold := generator.rng.f64_in_range(0.0, total_weight)!
		mut cumulative := 0.0
		mut selected_index := -1
		for index, weight in remaining {
			if weight <= 0 {
				continue
			}
			cumulative += weight
			if threshold < cumulative {
				selected_index = index
				break
			}
		}
		if selected_index < 0 {
			for index, weight in remaining {
				if weight > 0 {
					selected_index = index
				}
			}
		}
		selected[draw] = population.get_nth(selected_index)
		if !replace {
			total_weight -= remaining[selected_index]
			remaining[selected_index] = 0
		}
	}
	return from_1d[T](selected)
}

// permutation returns the integers in [0, size) in a seeded random order.
pub fn (mut generator RandomGenerator) permutation(size int) !&Tensor[int] {
	if size < 0 {
		return error('permutation: size must be non-negative')
	}
	mut values := []int{len: size}
	for i in 0 .. size {
		values[i] = i
	}
	for i := size - 1; i > 0; i-- {
		j := int(generator.rng.f64_in_range(0.0, f64(i + 1))!)
		values[i], values[j] = values[j], values[i]
	}
	return from_1d[int](values)
}

// permutation_tensor returns a copy with rows shuffled along the first axis,
// matching NumPy's permutation behavior for an N-dimensional array.
pub fn (mut generator RandomGenerator) permutation_tensor[T](input &Tensor[T]) !&Tensor[T] {
	if input.rank() == 0 {
		return error('permutation_tensor: input must have at least one dimension')
	}
	indices := generator.permutation(input.shape[0])!
	return input.take(indices.to_array(), 0)
}

// gamma returns samples from a Gamma distribution using this generator's
// independent stream. `alpha` is the shape and `scale` is the scale parameter.
pub fn (mut generator RandomGenerator) gamma(alpha f64, scale f64, shape []int) !&Tensor[f64] {
	validate_gamma_parameters(alpha, scale)!
	mut rng := generator.rng
	mut values := []f64{len: size_from_shape(shape)}
	for i in 0 .. values.len {
		values[i] = sample_gamma(alpha, scale, mut rng)!
	}
	return from_array[f64](values, shape)
}

fn validate_gamma_parameters(alpha f64, scale f64) ! {
	if alpha <= 0 || scale <= 0 || math.is_nan(alpha) || math.is_inf(alpha, 0)
		|| math.is_nan(scale) || math.is_inf(scale, 0) {
		return error('gamma: alpha and scale must be finite and positive')
	}
}

fn sample_gamma(alpha f64, scale f64, mut rng &rand.PRNG) !f64 {
	shape := if alpha < 1 { alpha + 1 } else { alpha }
	d := shape - 1.0 / 3.0
	c := 1.0 / math.sqrt(9 * d)
	for _ in 0 .. 10000 {
		x := rng.normal(config.NormalConfigStruct{})!
		if 1 + c * x <= 0 {
			continue
		}
		u := rng.f64_in_range(0.0, 1.0)!
		mut sample, accepted := gamma_candidate(d, c, x, u)
		if accepted {
			if alpha < 1 {
				sample *= math.pow(rng.f64_in_range(0.0, 1.0)!, 1 / alpha)
			}
			return sample * scale
		}
	}
	return error('gamma: rejection sampler did not converge')
}

fn gamma_candidate(d f64, c f64, x f64, uniform_value f64) (f64, bool) {
	base := 1 + c * x
	if base <= 0 {
		return 0, false
	}
	v := base * base * base
	if uniform_value < 1 - 0.0331 * x * x * x * x
		|| math.log(uniform_value) < 0.5 * x * x + d * (1 - v + math.log(v)) {
		return d * v, true
	}
	return 0, false
}

// beta returns samples from a Beta distribution using this generator's
// independent stream. `alpha` and `beta` are the two positive shape parameters.
pub fn (mut generator RandomGenerator) beta(alpha f64, beta f64, shape []int) !&Tensor[f64] {
	validate_gamma_parameters(alpha, 1)!
	validate_gamma_parameters(beta, 1)!
	mut rng := generator.rng
	mut values := []f64{len: size_from_shape(shape)}
	for i in 0 .. values.len {
		x := sample_gamma(alpha, 1, mut rng)!
		y := sample_gamma(beta, 1, mut rng)!
		if x + y == 0 {
			return error('beta: sampled gamma values underflowed to zero')
		}
		values[i] = x / (x + y)
	}
	return from_array[f64](values, shape)
}

// dirichlet returns samples from a Dirichlet distribution. The final output
// axes match the concentration tensor's shape; sample_shape is prepended.
pub fn (mut generator RandomGenerator) dirichlet[T](alpha &Tensor[T], sample_shape []int) !&Tensor[f64] {
	if alpha.rank() != 1 || alpha.size == 0 {
		return error('dirichlet: concentration parameters must be a non-empty vector')
	}
	for dimension in sample_shape {
		if dimension < 0 {
			return error('dirichlet: sample dimensions must be non-negative')
		}
	}
	mut concentrations := []f64{len: alpha.size}
	for i in 0 .. alpha.size {
		value := td(alpha.get_nth(i)).f64()
		if value <= 0 || math.is_nan(value) || math.is_inf(value, 0) {
			return error('dirichlet: concentration parameters must be finite and positive')
		}
		concentrations[i] = value
	}
	mut output_shape := sample_shape.clone()
	output_shape << alpha.size
	sample_count := size_from_shape(sample_shape)
	mut values := []f64{len: sample_count * alpha.size}
	mut rng := generator.rng
	for sample in 0 .. sample_count {
		start := sample * alpha.size
		mut total := 0.0
		for category, concentration in concentrations {
			values[start + category] = sample_gamma(concentration, 1.0, mut rng)!
			total += values[start + category]
		}
		if total == 0 || math.is_inf(total, 0) {
			return error('dirichlet: sampled gamma values could not be normalized')
		}
		for category in 0 .. alpha.size {
			values[start + category] /= total
		}
	}
	return from_array[f64](values, output_shape)
}

// bernoulli returns a tensor of bernoulli random variables.
pub fn bernoulli[T](prob f64, shape []int, params TensorData) &Tensor[T] {
	mut t := zeros[T](shape, params)
	mut iter := t.iterator()
	for {
		_, i := iter.next() or { break }
		rand_value := cast[T](rand.bernoulli(prob) or { 0.0 })
		t.set(i, rand_value)
	}
	return t
}

// binomial returns a tensor of binomial random variables.
pub fn binomial[T](n int, prob f64, shape []int, params TensorData) !&Tensor[T] {
	mut t := zeros[T](shape, params)
	mut iter := t.iterator()
	for {
		_, i := iter.next() or { break }
		rand_value := cast[T](rand.binomial(n, prob) or { return err })
		t.set(i, rand_value)
	}
	return t
}

// exponential returns a tensor of exponential random variables.
pub fn exponential[T](lambda f64, shape []int, params TensorData) &Tensor[T] {
	mut t := zeros[T](shape, params)
	mut iter := t.iterator()
	for {
		_, i := iter.next() or { break }
		rand_value := cast[T](rand.exponential(lambda))
		t.set(i, rand_value)
	}
	return t
}

// geometric returns the number of Bernoulli trials needed for the first
// success, sampled from V's global random stream. Results start at 1.
pub fn geometric(probability f64, shape []int, params TensorData) !&Tensor[int] {
	validate_geometric_probability(probability)!
	mut result := zeros[int](shape, params)
	for i in 0 .. result.size {
		u := rand.f64_in_range(0.0, 1.0)!
		result.set_nth(i, geometric_sample(probability, u))
	}
	return result
}

fn validate_geometric_probability(probability f64) ! {
	if probability <= 0 || probability > 1 || math.is_nan(probability) || math.is_inf(probability,
		0) {
		return error('geometric: probability must be finite and in (0, 1]')
	}
}

fn geometric_sample(probability f64, uniform_value f64) int {
	if probability == 1 {
		return 1
	}
	return int(math.floor(math.log(1 - uniform_value) / math.log(1 - probability))) + 1
}

// poisson returns integer event counts from VTL's global random stream.
pub fn poisson(lambda f64, shape []int, params TensorData) !&Tensor[int] {
	validate_poisson_rate(lambda)!
	mut result := zeros[int](shape, params)
	for i in 0 .. result.size {
		result.set_nth(i, sample_poisson_global(lambda)!)
	}
	return result
}

// weibull returns samples from the unit-scale Weibull distribution using
// VTL's global random stream.
pub fn weibull(shape_parameter f64, shape []int, params TensorData) !&Tensor[f64] {
	if shape_parameter <= 0 || math.is_nan(shape_parameter) || math.is_inf(shape_parameter, 0) {
		return error('weibull: shape parameter must be finite and positive')
	}
	mut result := zeros[f64](shape, params)
	for i in 0 .. result.size {
		u := rand.f64_in_range(0.0, 1.0)!
		result.set_nth(i, math.pow(-math.log(1.0 - u), 1.0 / shape_parameter))
	}
	return result
}

// gamma returns samples from a Gamma distribution on VTL's global random
// stream. `alpha` is the shape and `scale` is the scale parameter.
pub fn gamma(alpha f64, scale f64, shape []int, params TensorData) !&Tensor[f64] {
	validate_gamma_parameters(alpha, scale)!
	mut result := zeros[f64](shape, params)
	for i in 0 .. result.size {
		result.set_nth(i, sample_gamma_global(alpha, scale)!)
	}
	return result
}

// beta returns samples from the Beta distribution using VTL's global stream.
pub fn beta(alpha f64, beta f64, shape []int, params TensorData) !&Tensor[f64] {
	validate_gamma_parameters(alpha, 1)!
	validate_gamma_parameters(beta, 1)!
	mut result := zeros[f64](shape, params)
	for i in 0 .. result.size {
		x := sample_gamma_global(alpha, 1)!
		y := sample_gamma_global(beta, 1)!
		if x + y == 0 {
			return error('beta: sampled gamma values underflowed to zero')
		}
		result.set_nth(i, x / (x + y))
	}
	return result
}

// chi_square returns samples from a chi-square distribution using VTL's
// global random stream.
pub fn chi_square(degrees_of_freedom f64, shape []int, params TensorData) !&Tensor[f64] {
	validate_degrees_of_freedom(degrees_of_freedom, 'chi_square')!
	mut result := zeros[f64](shape, params)
	for i in 0 .. result.size {
		result.set_nth(i, sample_gamma_global(degrees_of_freedom / 2, 2)!)
	}
	return result
}

// student_t returns samples from the standard Student's t distribution using
// VTL's global random stream.
pub fn student_t(degrees_of_freedom f64, shape []int, params TensorData) !&Tensor[f64] {
	validate_degrees_of_freedom(degrees_of_freedom, 'student_t')!
	mut result := zeros[f64](shape, params)
	for i in 0 .. result.size {
		normal_sample := rand.normal(config.NormalConfigStruct{})!
		chi_square_sample := sample_gamma_global(degrees_of_freedom / 2, 2)!
		if chi_square_sample <= 0 {
			return error('student_t: sampled chi-square value underflowed to zero')
		}
		result.set_nth(i, normal_sample / math.sqrt(chi_square_sample / degrees_of_freedom))
	}
	return result
}

// f_distribution returns samples from the F distribution using VTL's global
// random stream.
pub fn f_distribution(numerator_df f64, denominator_df f64, shape []int, params TensorData) !&Tensor[f64] {
	validate_degrees_of_freedom(numerator_df, 'f_distribution')!
	validate_degrees_of_freedom(denominator_df, 'f_distribution')!
	mut result := zeros[f64](shape, params)
	for i in 0 .. result.size {
		numerator := sample_gamma_global(numerator_df / 2, 2)!
		denominator := sample_gamma_global(denominator_df / 2, 2)!
		if denominator <= 0 {
			return error('f_distribution: sampled denominator underflowed to zero')
		}
		result.set_nth(i, (numerator / numerator_df) / (denominator / denominator_df))
	}
	return result
}

fn sample_gamma_global(alpha f64, scale f64) !f64 {
	shape := if alpha < 1 { alpha + 1 } else { alpha }
	d := shape - 1.0 / 3.0
	c := 1.0 / math.sqrt(9 * d)
	for _ in 0 .. 10000 {
		x := rand.normal(config.NormalConfigStruct{})!
		if 1 + c * x <= 0 {
			continue
		}
		u := rand.f64_in_range(0.0, 1.0)!
		mut sample, accepted := gamma_candidate(d, c, x, u)
		if accepted {
			if alpha < 1 {
				sample *= math.pow(rand.f64_in_range(0.0, 1.0)!, 1 / alpha)
			}
			return sample * scale
		}
	}
	return error('gamma: rejection sampler did not converge')
}

fn validate_poisson_rate(lambda f64) ! {
	if lambda < 0 || math.is_nan(lambda) || math.is_inf(lambda, 0) || lambda >= f64(max_int) {
		return error('poisson: lambda must be finite, non-negative, and fit in an int')
	}
}

fn sample_poisson_global(lambda f64) !int {
	if lambda == 0 {
		return 0
	}
	if lambda < 10 {
		limit := math.exp(-lambda)
		mut product := 1.0
		mut count := 0
		for product > limit {
			count++
			product *= rand.f64_in_range(0.0, 1.0)!
		}
		return count - 1
	}
	ptrs := poisson_ptrs_config(lambda)
	for _ in 0 .. 10000 {
		u := rand.f64_in_range(0.0, 1.0)!
		v := rand.f64_in_range(0.0, 1.0)!
		candidate, accepted := sample_poisson_ptrs_candidate(ptrs, u, v)
		if accepted {
			return candidate
		}
	}
	return error('poisson: rejection sampler did not converge')
}

struct PoissonPtrsConfig {
	lambda           f64
	a                f64
	b                f64
	inverse_alpha    f64
	quick_acceptance f64
}

fn poisson_ptrs_config(lambda f64) PoissonPtrsConfig {
	sqrt_lambda := math.sqrt(lambda)
	b := 0.931 + 2.53 * sqrt_lambda
	a := -0.059 + 0.02483 * b
	return PoissonPtrsConfig{
		lambda:           lambda
		a:                a
		b:                b
		inverse_alpha:    1.1239 + 1.1328 / (b - 3.4)
		quick_acceptance: 0.9277 - 3.6224 / (b - 2)
	}
}

fn sample_poisson_ptrs_candidate(ptrs PoissonPtrsConfig, uniform_value f64, threshold f64) (int, bool) {
	u := uniform_value - 0.5
	window := 0.5 - math.abs(u)
	candidate := math.floor((2 * ptrs.a / window + ptrs.b) * u + ptrs.lambda + 0.43)
	if candidate < 0 || candidate >= f64(max_int) {
		return 0, false
	}
	if window >= 0.07 && threshold <= ptrs.quick_acceptance {
		return int(candidate), true
	}
	if window < 0.013 && threshold > window {
		return 0, false
	}
	left := math.log(threshold * ptrs.inverse_alpha / (ptrs.a / (window * window) + ptrs.b))
	right := -ptrs.lambda + candidate * math.log(ptrs.lambda) - math.log_factorial(candidate)
	return int(candidate), left <= right
}

// NormalTensorData is the data for a normal distribution.

// NormalTensorData defines a public data structure for this module.

// NormalTensorData defines a public data structure for this module.
@[params]
pub struct NormalTensorData {
	TensorData
	config.NormalConfigStruct
}

// normal returns a tensor of normal random variables.
pub fn normal[T](shape []int, params NormalTensorData) &Tensor[T] {
	mut t := zeros[T](shape, memory: params.memory)
	mut iter := t.iterator()
	for {
		_, i := iter.next() or { break }
		rand_value := cast[T](rand.normal(mu: params.mu, sigma: params.sigma) or { math.nan() })
		t.set(i, rand_value)
	}
	return t
}

// random returns a new Tensor of given shape and type, initialized
// with random numbers between a given min and max value
pub fn random[T](min T, max T, shape []int, params TensorData) &Tensor[T] {
	mut t := zeros[T](shape, params)
	mut iter := t.iterator()
	for {
		_, i := iter.next() or { break }
		rand_value := random_in_range[T](min, max)
		t.set(i, rand_value)
	}
	return t
}

// random_seed resets VTL's global random generator for reproducible results.
// The seed affects random tensor constructors and model parameter
// initialization that use V's default random generator.
pub fn random_seed(i int) {
	rand.seed([u32(i), 0])
}

fn random_in_range[T](min T, max T) T {
	$if T is u16 {
		return u16(rand.int_in_range(int(min), int(max)) or { u16(min) })
	}
	$if T is u8 {
		return u8(rand.int_in_range(int(min), int(max)) or { int(min) })
	}
	$if T is u32 {
		return rand.u32_in_range(min, max) or { min }
	}
	$if T is u64 {
		return rand.u64_in_range(min, max) or { min }
	}
	$if T is i8 {
		return i8(rand.int_in_range(int(min), int(max)) or { i8(min) })
	}
	$if T is i16 {
		return i16(rand.int_in_range(int(min), int(max)) or { int(min) })
	}
	$if T is i32 {
		return i32(rand.int_in_range(int(min), int(max)) or { int(min) })
	}
	$if T is int {
		return rand.int_in_range(min, max) or { min }
	}
	$if T is i64 {
		return rand.i64_in_range(min, max) or { min }
	}
	$if T is f32 {
		return rand.f32_in_range(min, max) or { min }
	}
	$if T is f64 {
		return rand.f64_in_range(min, max) or { min }
	}
	return min
}
