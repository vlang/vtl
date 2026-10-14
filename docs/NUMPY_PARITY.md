# NumPy and Arraymancer Feature Parity

VTL's long-term goal is to provide a more complete and more performant tensor
and machine-learning toolkit than Arraymancer, then grow beyond NumPy for the
array operations that make sense in V. This is a capability roadmap, not a
claim of current parity. Every checked item must be backed by public API,
tests, documentation, and examples where users need them.

## Open implementation issues

| Area | Issue |
|------|-------|
| Numeric promotion and tensor creation | [#717](https://github.com/vlang/vtl/issues/717), [#718](https://github.com/vlang/vtl/issues/718) |
| Ufunc and elementwise math coverage | [#777](https://github.com/vlang/vtl/issues/777) |
| Copy/view rules and indexing bounds | [#719](https://github.com/vlang/vtl/issues/719), [#720](https://github.com/vlang/vtl/issues/720) |
| Reductions and complex support | [#721](https://github.com/vlang/vtl/issues/721), [#722](https://github.com/vlang/vtl/issues/722) |
| Random streams and FFT parity | [#723](https://github.com/vlang/vtl/issues/723), [#724](https://github.com/vlang/vtl/issues/724) |
| Array I/O formats | [#725][i725], [#726][i726], [#779][i779] |
| Masked autograd and structured data | [#727](https://github.com/vlang/vtl/issues/727), [#728](https://github.com/vlang/vtl/issues/728) |
| Backend benchmarks and learning examples | [#729](https://github.com/vlang/vtl/issues/729), [#730](https://github.com/vlang/vtl/issues/730) |
| OpenCL compute and neural-network dispatch | [#750](https://github.com/vlang/vtl/issues/750) |
| Sparse tensor operations | [#746](https://github.com/vlang/vtl/issues/746) |
| Non-numeric dtypes and array interoperability | [#747](https://github.com/vlang/vtl/issues/747), [#748](https://github.com/vlang/vtl/issues/748) |
| NumPy polynomial families and fitting | [#773](https://github.com/vlang/vtl/issues/773) |

For neural-network and framework-level gaps relative to PyTorch, see the
[PyTorch parity tracker](./PYTORCH_PARITY.md).

## Current strengths

- Generic N-dimensional tensors, slicing, broadcasting, reshape, transpose,
  axis swapping, and view-preserving `moveaxis`/`rollaxis`.
- Elementwise math, reductions, mapping, stacking, splitting, random values,
  statistics, and linear algebra through VSL.
- Reverse-mode autograd, neural-network layers and losses, optimizers,
  datasets, CPU support, and selected CUDA/Vulkan operations. OpenCL/VCL
  currently covers tensor transfer and an `add_scalar` smoke example; VTL
  OpenCL compute and neural-network dispatch remain incomplete (see the
  [backend roadmap](../ROADMAP.md#phase-5--opencl-backend)).
- Tutorials and runnable examples for tensors, linear algebra, autograd,
  training, and GPU backends.

## NumPy domains

### Array creation

**Current:** zeros, ones, full, eye, range, sequence, arrays, and Vandermonde
matrices; `arange` with start/stop/step; endpoint-aware `linspace`; configurable
`logspace`; 2-D `meshgrid` and N-D `meshgrid_n` with `xy` and `ij` indexing;
dense stacked coordinates through `indices` and broadcastable per-axis sparse
coordinates through `indices_sparse`. VTL also provides `one_hot` for ML labels,
appending a class dimension and validating class bounds.

**Remaining:** validate edge cases and make dtype/device options consistent.

### Shape and manipulation

**Current:** reshape, transpose, squeeze/expand, move/roll axes, N-D
`diagonal` views over two selected axes with NumPy-style offset and output
shape, concatenate/stack/split, `ravel`, copying `flatten`, `flip`, circular
`roll` with NumPy's flattened default, scalar shifts over one or multiple axes,
paired shifts per axis, `repeat`, `repeat_axis`, block `tile`, `rot90`,
constant/edge/wrap/reflect/symmetric
padding, and stable axis-wise `sort`/`argsort`.
`reshape` follows row-major logical order, shares storage for row-major
contiguous input, and copies when another layout cannot be reshaped as that
view; tests cover transposed and column-major values as well as storage sharing.
Top-level `tril` and `triu` copy N-D inputs and apply NumPy's diagonal offset
to each trailing matrix.

**Current:** `broadcast_to`, `broadcast2`, `broadcast3`, and `broadcast_n`
create zero-copy views, validate incompatible shapes, and follow NumPy's
zero-dimension rules. Tests cover empty inputs, rank mismatches, and invalid
target dimensions.

**Remaining:** audit copy-versus-view behavior across manipulation APIs beyond
the tested reshape, transpose, indexing, and diagonal cases.
VTL diagonal views share writable storage; NumPy exposes its diagonal views as
read-only by default.

### Indexing and set operations

**Current:** integer indexing and slices; broadcasted coordinate-array
`advanced_index`; mixed `mixed_index` selectors for coordinate arrays, scalar
indices, slices, ellipsis, and new axes; sorted `unique` and its counts,
inverse, and first-index variants;
`unique_axis`; `digitize`; ascending and descending `searchsorted`; global,
single-axis, and multiple-axis `count_nonzero`; `argwhere` and per-axis `nonzero`;
`take`, `take_nd`,
`take_flat`, and their `raise`/`wrap`/`clip` mode variants;
`take_along_axis` with broadcasting outside the selected axis;
`put_along_axis` and `scatter_add`; lookup and DataLoader gathers; broadcastable
`masked_select`, `masked_fill`, and `masked_scatter`; leading-dimension boolean-mask indexing via
`boolean_index`, which preserves trailing dimensions like NumPy `array[mask]`.
`Tensor.put` and `put_with_mode` update row-major logical flat indices with
NumPy's `raise`, `wrap`, and `clip` modes; short value tensors repeat, and
validation completes before mutation. `Tensor.putmask` performs in-place
row-major masked updates with an exact-shape mask and cycles values by each
destination flat index, matching NumPy.
`select` applies ordered broadcastable conditions to choice tensors; the first
true condition wins and `default` fills positions with no match, matching
`numpy.select` when all choices and the default share one V element type.
`ravel_multi_index` broadcasts coordinate tensors and supports C/F flattening,
global `raise`/`wrap`/`clip` bounds modes, and per-axis bounds modes;
`unravel_index` returns one coordinate tensor per dimension in the requested
order. The indexing tutorial and runnable `multi_index` example cover both
orders and a bounds-mode case.

`Variable.take`, `Variable.take_nd`, `Variable.take_flat`,
`Variable.take_along_axis`, `Variable.boolean_index`, and
`Variable.masked_select` propagate gradients by scattering them to selected
source positions, including accumulation for repeated indices and reduction of
broadcast source dimensions. `Variable.masked_fill` zeros gradients at replaced
input positions.
`Variable.scatter_add` returns an updated copy and propagates gradients to its
source and update values, including repeated destination indices.
`Variable.put_along_axis` follows last-write-wins semantics and routes gradients
only to the final update at each destination.
`Variable.put` and `Variable.put_with_mode` provide the same gradient behavior
for flat row-major indices, including repeated short updates and `raise`, `wrap`,
and `clip` index modes. Indices are snapshotted during the forward pass, and
repeated uses of one update value accumulate into its gradient.
`Variable.putmask` returns an updated copy and routes gradients to unmasked input
positions and to update values selected by each true destination's flat index.
`Variable.masked_scatter` broadcasts its mask, consumes source values in order
for true destinations, and routes gradients to the input and consumed source
prefix.
`Variable.slice` and `Variable.slice_hilo` propagate gradients through integer
indices, range views, and positive steps. `Variable.roll` and
`Variable.roll_shifts` support flattened and axis-wise circular shifts and
apply the inverse shifts during backward. `Variable.rot90`, `rot90_k`, and
`rot90_axes` propagate gradients through the inverse rotation. `Variable.flip`
propagates gradients by flipping them over the same axes. `Variable.repeat` and
`Variable.repeat_axis` sum gradients from repeated copies back to each source
element. `Variable.tile` sums gradients from repeated blocks. The tensor axis
manipulation example shows rotation, flip, repeat, tile, and their
corresponding gradient paths. `Variable.pad` routes gradients through constant,
edge, wrap, reflect, and symmetric padding modes.
`Variable.sum` and `Variable.mean` reduce all elements to a one-element tensor
and propagate gradients to every input element. Their `*_along_axis` variants
support one-axis sum and mean, with either squeezed or retained dimensions.

**Remaining:** continue expanding NumPy-compatible bounds semantics across
the other indexing APIs. `take_along_axis` keeps its NumPy behavior of raising
on invalid indices. Tensor indexed writes remain in-place and
non-differentiable; use the `Variable` APIs when gradients are required.

Axis-wise `argmax` and `argmin` retain a length-one axis for compatibility;
`argmax_axis_squeeze` and `argmin_axis_squeeze` provide NumPy's default
axis-removed output shape. VTL represents scalar-shaped reductions as shape
`[1]`. `argmax_flat()` and `argmin_flat()` provide NumPy-style flattened
indices; standard arg reductions return the first NaN index if present.
`nanargmax()` and `nanargmin()` skip NaNs. Their `*_axis(axis, keepdims)`
variants reduce one axis and support strided views. NaN arg reductions report
an error for empty inputs, empty reduced axes, or all-NaN slices.
Value extrema through `max_axis` and `min_axis` preserve the reduced axis,
propagate NaNs, support strided views, and report an error when the reduced
axis is empty. Use the corresponding `*_axis_squeeze` methods for a
NumPy-style axis-removed output shape.

### Math and ufuncs

**Current:** broad elementwise math and broadcasting; same-dtype tensor `+`,
`-`, `*`, and `/` operators; `clip` and fused
`clip_tensor`; broadcast-aware `where`; elementwise `is_nan`, `is_inf`, and
`is_finite`; `isclose`, `allclose`, `heaviside`, `sign`, `diff`, and numerical
gradients along selected axes; elementwise logical AND/OR/XOR/NOT; integer
bitwise AND/OR/XOR/invert and scalar left/right shifts. Gradients support
uniform spacing or monotonic non-uniform coordinates and first/second-order
boundary differences. `add_promoted`, `subtract_promoted`, and
`multiply_promoted` accept mixed real numeric dtypes, broadcast their inputs,
and validate an explicitly selected output dtype against `promote_types`.
`divide_promoted` performs NumPy-style true division for mixed real numeric and
boolean tensors, broadcasts inputs, and returns `float32` only when the
promoted input dtype is `float32`; other supported combinations return
`float64`.
`remainder_promoted` performs NumPy-style remainder with mixed integer and
floating-point dtypes and broadcasting. Nonzero results follow the divisor's
sign; signed zero follows the divisor; integer zero divisors produce zero and
floating-point zero divisors produce NaN. Floating-point remainder is a tensor
operation and does not yet propagate gradients through `Variable`. See the
[implementation](../math_promoted.v), [focused tests](../tests/core/math_promoted_remainder_test.v),
and [dtype conversion tutorial](./TUTORIAL_FIRST_STEPS.md#numeric-dtype-conversion).
`round_decimals` rounds real floating-point tensors to positive or negative
decimal places using ties-to-even, matching NumPy `round` for the supported
same-dtype inputs; integer tensors are copied unchanged. VTL preserves finite
inputs rather than overflowing when an extreme positive decimal position
would overflow its intermediate scale.
`nan_to_num` replaces NaN with zero and infinities with finite type limits for
f32/f64 tensors and replaces non-finite real and imaginary components
independently for complex128 tensors, with optional custom replacements.
Integer tensors are copied. For real floating-point tensors,
`Variable.nan_to_num` also
propagates gradients through finite inputs and returns zero gradients for
replaced NaN/infinite inputs, matching PyTorch.
See [NumPy's nan_to_num reference](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
for the component-wise complex behavior.
`Tensor.modf` returns fractional then integral parts, matching NumPy's output
order, and `Tensor.frexp` returns floating-point mantissas with integer base-2
exponents. The `ldexp` function reconstructs values and broadcasts mantissas
against integer exponent tensors. These decomposition ufuncs support f32/f64;
see the [math ufunc tutorial](./TUTORIAL_MATH_UFUNCS.md) and
[`math_decomposition` example](../examples/math_decomposition/README.md).
Elementwise `max`/`min` match NumPy's NaN-propagating `maximum`/`minimum`, and
`fmax`/`fmin` ignore a single NaN operand. These operations broadcast and
support boolean, integer, floating-point, and complex tensors; floating-point
zero ties follow each NumPy ufunc's sign behavior.

Unary transcendental operations (`exp`, `sin`, `log`, and 20 more) use a
contiguous fast path that calls libm directly per element for f32/f64
(f32 is evaluated in f64 and rounded once), avoiding the previous
per-element closure and sum-type boxing. On this machine (1M f64, `-prod`)
`exp` improved from 12.1 ms to 4.7 ms per iteration (NumPy: ~3.3 ms) and
`sin` measures 15.5 ms against NumPy's ~12.8 ms.

**Remaining:** audit unary/binary function families and complete the missing
NumPy ufunc coverage in [#777](https://github.com/vlang/vtl/issues/777). Add
promoted arithmetic for the remaining operations and infer output dtypes
automatically for mixed tensor and weak scalar operands in
[#717](https://github.com/vlang/vtl/issues/717).

### Reductions and statistics

**Current:** single-axis and multi-axis sum/product, NaN-aware sum/product/
min/max, and logical all/any with `keepdims`; weighted average globally, over
one axis, and over multiple axes with optional `keepdims`; scalar
population/sample variance and standard deviation; single-axis and multi-axis
mean/variance/std with explicit `keepdims` and NaN-aware forms; explicit
accumulator dtypes for global, single-axis, and multi-axis regular sum/product
and NaN-aware sum/product; axis-wise
`trapezoid`; all thirteen NumPy quantile estimators via `QuantileMethod`, with
global, axis, multi-axis, multi-quantile, percentile, and NaN-aware forms; axis arg
reductions and squeezed min/max/argmin/argmax results; median and NaN-aware
median helpers over one or multiple axes with explicit `keepdims`; multi-axis min/max;
weighted inverted-CDF quantile, multi-quantile, percentile, and multi-percentile
forms globally and along one or several axes, with same-shape or reduced-shape
weights, including NaN-aware forms that discard weights for NaN observations;
single-axis weighted multi-quantile and multi-percentile APIs also provide
`*_keepdims` variants that retain the reduced axis after the leading quantile
dimension;
multi-axis weighted multi-quantile and multi-percentile APIs provide the same
retained-axis behavior;
multi-axis quantile and percentile convenience APIs default to NumPy's linear
estimator, alongside forms that expose all thirteen estimator methods;
multi-quantile multi-axis forms can also retain reduced dimensions with
`keepdims`; single-axis multi-quantile forms provide the same retained-shape
behavior;
one- through N-dimensional histograms with inferred and custom edges, weighted
counts, densities normalized by each bin's volume, and multiple one-dimensional
bin-selection rules; peak-to-peak range via `stats.ptp` globally and across one
or several axes with optional `keepdims` for integer and floating-point input;
integer/weighted `bincount`.

Global and tensor-axis `min`/`max` propagate NaNs; their `nanmin`/`nanmax`
variants skip NaNs and return NaN when a reduction has no valid values.

`histogramdd` matches sample and bin ordering and density normalization for the
covered `(N, D)` tensor API. Unweighted VTL counts use `int` storage, while
NumPy returns `float64`; this output-dtype difference remains explicit.

**Current:** explicit accumulator/output dtypes are available for global,
single-axis, and multi-axis sum and product reductions. Cumulative sum and
product along one axis also provide explicit output dtype selection through
`cumsum_as` and `cumprod_as`.
`mean_as`, `mean_along_axis_as`, and `mean_along_axes_as` likewise select the
accumulator and output type for ordinary means. `variance_as` and `std_as`,
with their single-axis and multi-axis forms, provide the same control for
ordinary variance and standard deviation. Existing APIs remain available with
their original return types. NaN-aware mean, variance, and standard deviation
also provide typed global, single-axis, and multi-axis APIs; empty or all-NaN
floating-point slices produce NaN.
`stats.sum_accurate` provides an opt-in Neumaier-compensated global sum for
floating-point tensors. `Tensor.cumsum_flat` and `Tensor.cumprod_flat` provide
NumPy's flattened `axis=None` cumulative behavior, with explicit accumulator
types through `cumsum_flat_as` and `cumprod_flat_as`; their `Variable`
counterparts propagate gradients through the flattened order.

**Remaining:** add `keepdims` options consistently across remaining
reductions, extend accumulator dtype controls to other reduction families, and
broaden reduction coverage.

Logical `all_axis`/`any_axis` and multi-axis `all_axes`/`any_axes` reductions
return `bool` tensors; empty reduced dimensions return the logical identities
(true and false respectively). Empty axis lists perform elementwise truth
conversion, matching NumPy's `axis=()` behavior.

### Reduction behavior matrix

This matrix records the current public contracts, including legacy differences
that callers must account for. “Empty reduction” means the reduced input has no
elements; an empty output caused only by an unreduced dimension remains empty.

| Family | Axes / `keepdims` | Output type | NaNs | Empty reduction |
|--------|-----------------|-------------|------|-----------------|
| `sum` / `product` | G/A/M; KD on A/M; A=() is identity | T; `_as` selects U | Propagate | Identities 0 / 1 |
| `nansum` / `nanprod` | G/A/M; KD on A/M | f64; `_as` selects U | Skip | Identities 0 / 1 |
| `min` / `max` | G/A/M; A retains by default; M takes KD | T | Propagate first NaN | G returns 0; empty reduced A errors |
| `nanmin` / `nanmax` | G/A/M; retained and squeezed A forms | f64 | Skip | NaN if empty or no valid values |
| `mean` / `variance` / `std` | G/A/M; KD on tensor A/M forms | mean T; var/std f64; `_as` selects U | Propagate | See moments details below |
| `nanmean` / `nanvar` / `nanstd` | G/A/M; KD on tensor A/M forms | f64; `_as` selects U | Skip | NaN if empty, invalid, or under-sampled |
| `argmin` / `argmax` | G flattened index; A retained or squeezed | int | First NaN; `nanarg*` skips | Empty/no-valid reductions error |
| `median` / quantiles | G/A/M; KD forms; multi-q prepends q axis | f64 | Ordinary forms propagate; `nan*` skips | Median NaN; quantile errors |
| Weighted average / quantiles | G/A/M; KD; multi-q adds q axis | f64 | Average propagates; `nan` skips | Empty/zero-weight errors |
| `all` / `any` | G/A/M; KD; A=() is elementwise | bool | NaN is truthy | Identities true / false |
| `cumsum` / `cumprod` | A preserves shape; flat returns 1-D; no KD | T; `_as` selects U | Propagates forward | Empty tensor |
| `ptp` | G/A/M; KD on tensor forms | T | Propagate | Empty reduced extent errors |

Legend: G=global, A=one axis, M=multiple axes, KD=`keepdims`, T=input type,
U=explicit output/accumulator type, q=number of requested quantiles.

For moments, legacy global `mean` returns the additive identity on an empty
real tensor; `mean_as[T, f64]` returns NaN. Global variance/std error when the
observation count does not exceed `ddof`; axis variance/std return NaN for
under-sampled slices. NaN-aware moments return NaN for empty/all-NaN inputs or
slices and when valid observations do not exceed `ddof`.

Axis `argmin`/`argmax` keep the reduced dimension in their default forms;
`*_squeeze` removes it. Ordinary quantile forms propagate NaNs in covered tests;
ordinary median propagates NaNs globally and for axis/multi-axis slices;
`nan*` forms skip them. Empty median returns NaN, while empty quantile/percentile
calls error. Weighted average errors on empty inputs or zero total weight;
weighted quantiles use the inverted-CDF method and error if valid observations
have zero total weight.

The global empty-input results for `mean`, `min`, and `max` are compatibility
differences in the legacy generic APIs. `mean_as[T, f64]` returns NaN for an
empty input, while `nanmean`, `nanmin`, and `nanmax` return NaN when they have
no valid values. Regression coverage lives in
[`axis_reductions_test.v`](../tests/core/axis_reductions_test.v),
[`multi_axis_reductions_test.v`](../tests/core/multi_axis_reductions_test.v),
[`logical_reduction_test.v`](../tests/core/logical_reduction_test.v),
[`reduction_test.v`](../tests/core/reduction_test.v), and the
[`stats` tests](../stats/), including
[`median_axes_test.v`](../stats/median_axes_test.v).

### Linear algebra

**Current:** NumPy-style vector/matrix `matmul` promotion and batched
broadcasting for real and complex128 tensors; N-D Kronecker products with
NumPy rank-promotion semantics; Einstein summation for labeled contractions,
repeated-label diagonals, multi-operand expressions, ellipses, broadcasting,
and complex128 values;
offset diagonal construction and extraction; solve, least squares,
QR/LU/Cholesky, pseudoinverse, trace, matrix norms,
flattened, single-axis, and tuple-axis p-norms, and covariance/correlation
matrices.

`la.vector_norm_axes(t, ord, axes, keepdims)` reduces any non-empty tuple of
unique axes in one operation. It accepts negative axes, preserves retained
dimensions when requested, and supports strided views. This covers the
multi-axis vector norm use case in NumPy's
[`numpy.linalg.vector_norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.vector_norm.html).
See the [vector norm example](../examples/vector_norm/main.v).

`la.matrix_norm` computes NumPy's eight standard matrix norms for each trailing
matrix in an N-D tensor, with optional retained matrix dimensions.
`la.trace_axes` sums offset diagonals across selected axes and preserves the
remaining dimensions as a batch.
`la.svdvals` returns descending singular values for batched real matrices.
`la.svd` returns U, descending singular values, and V transpose for batched
real matrices, with NumPy-compatible full and reduced matrix shapes selected by
`full_matrices` (default `true`). The pure-V one-sided Jacobi implementation
supports rectangular inputs and rejects non-finite values.
Build with `-d vsl_lapack_lapacke` to dispatch SVD to the native LAPACKE
backend; the default build remains dependency-free and uses the pure-V path.
`la.solve_complex` solves complex128 vector and matrix right-hand sides with
partial pivoting and supports broadcast batch dimensions. As in NumPy 2.0, a
right-hand side is treated as a vector only when it has rank one. `la.inv_complex`
uses the same solver for batched complex matrix inverses; `la.det_complex`
computes complex determinants with pivoted elimination. Complex `matmul`, solve,
inverse, determinant, and least squares are currently supported. Complex
factorizations and eigenproblems remain future work.
`la.slogdet` returns determinant signs and log absolute determinants for batches,
including singular matrices and values whose determinants would overflow.
`la.slogdet_complex` returns unit-complex phases and log magnitudes for
complex128 matrix batches, including singular and empty matrices.
`la.matrix_power` uses exponentiation by squaring for batched integer powers and
partial-pivot inversion for negative exponents.
`la.multi_dot` uses dynamic programming to minimize the estimated scalar
multiplication count for a matrix chain, allowing vectors only at either end.
`la.cond` computes spectral and induced-norm condition numbers for stacked
square matrices, with singular spectral cases returning infinity or zero for
orders `2` and `-2` respectively.
`la.solve` supports vector and matrix right-hand sides, stacked systems, and
NumPy-style broadcasting of leading dimensions. `la.det` and `la.inv` operate
on stacks of square matrices; inverse results are checked against their input
layout and singular inputs report errors.
`la.eigh` and `la.eigvalsh` compute ascending eigenvalues for batches of real
symmetric matrices, with eigenvectors in columns and selectable upper/lower
triangle input. Inputs must be finite; the Jacobi method returns an error if
it does not converge within 100 sweeps. `la.eig` and `la.eigvals` handle batches
of finite real non-symmetric matrices and return complex eigenvalues; `la.eig`
also returns right eigenvectors in columns, including conjugate pairs.
`la.matrix_rank` computes one matrix's numerical rank with an explicit or
dtype-aware NumPy-style default tolerance; `la.matrix_rank_batch` preserves
leading batch dimensions.
`la.lstsq` preserves vector versus matrix solution shapes, computes numerical
rank from a relative singular-value cutoff, and returns squared residual sums
only when NumPy does. `la.lstsq_complex` accepts complex128 vector and matrix
right-hand sides and preserves solution shapes, numerical rank, singular values,
and residual conventions. It currently embeds the problem in real arithmetic,
using temporary matrices twice as large along each axis.

**Remaining:** add complex factorizations, complex-input general eigenproblems,
and complex SVD; benchmark realistic shapes for both the default and LAPACKE
backends.

### FFT

**Current:** 1-D, selected-axis, and N-D complex transforms for f32/f64; real
`rfft`/`rfftn` and inverses; reusable real plans; `fftfreq`/`rfftfreq` and
`fftshift`/`ifftshift`; NumPy's `backward`/`forward`/`ortho` normalization.
Dedicated `_f32` real APIs preserve f32 output; generic `rfft[f32]` retains its
f64 complex return type.

**Remaining:** benchmark large/strided arrays and complete the dtype-promotion
audit.

### Random

**Current:** global uniform, normal, Bernoulli, binomial, negative-binomial,
geometric, hypergeometric,
exponential, Gumbel, Laplace, Logistic, Cauchy, Wald (inverse Gaussian),
Pareto, Rayleigh, triangular, Poisson,
Weibull, lognormal, gamma, beta, Dirichlet, chi-square, Student's t, and F
tensors; `random_seed`;
independent seeded `RandomGenerator` streams for f64 uniform, normal, lognormal,
Gumbel, Laplace, Logistic, Cauchy, Wald, Pareto, Rayleigh, triangular, gamma, beta,
Dirichlet, exponential, Poisson, Weibull, chi-square, Student's t, F,
hypergeometric, negative-binomial, multivariate normal with positive-semidefinite
covariance matrices, and boolean Bernoulli; integer
uniform, binomial/geometric/multinomial/Poisson; uniform and weighted population choice
over flattened values or complete slices along a selected axis;
and integer, first-axis, or selected-axis tensor permutations.

**Remaining:** add other less common NumPy distributions and
define reproducibility across runtime versions.

### Input and output

**Current:** model serialization and dataset loaders; numeric CSV tensors;
typed `.npy` and named `.npz` I/O with mixed supported dtypes. `.npy` accepts
v1/v2/v3 headers, both byte orders, Fortran order, bool, f32/f64, complex128, and signed or unsigned
integer widths when the requested V type matches.
`.npz` read/write also supports complex128 members.

**Remaining:** support more text formats and dtypes, add memory-mapped arrays
and optional HDF5 interoperability, add independent NumPy fixtures, and
continue rejecting unsupported object/structured values. See
[#725](https://github.com/vlang/vtl/issues/725),
[#726](https://github.com/vlang/vtl/issues/726), and
[#779](https://github.com/vlang/vtl/issues/779).

### Data types

**Current:** V generic element types, `dtype()` introspection, and
`promote_types` for supported array dtypes. `math.complex.Complex` is supported
as `complex128` tensor storage, including `zeros`/`ones`, elementwise add,
subtract, multiply, and divide, plus NumPy-style `real`, `imag`, `conj`,
`absolute`, and `abs`
operations, exponentials/logarithms, square root, trigonometric and
hyperbolic functions, their common inverse families, phase/angle, and complex
NaN/infinity/finite predicates. Global sum, product,
and mean return complex scalars; single- and multi-axis sum/product return
complex tensors. Complex means along one or several axes return complex tensors;
complex variance and standard deviation return real tensors and support
`keepdims` and `ddof`. Promotion with a complex operand produces `complex128`. Explicit
`as_*` conversions cover real numeric and boolean types, preserve shape and
logical order, and truncate floating-point values when converting to integers.

**Remaining:** extend promoted arithmetic to all supported operations, add
weak scalar promotion and string conversion, extend complex support beyond
elementwise operations, `matmul`, Einstein contractions, `solve_complex`,
`inv_complex`, `det_complex`, `slogdet_complex`, and `lstsq_complex` to more
general linear algebra and random generation, add complex32, and define integer
overflow and structured-data limits.

### Masked and missing data

**Current:** `MaskedArray[T]` pairs values with a broadcastable boolean mask
(`true` means missing). It supports broadcasted arithmetic and comparisons
with mask union, scalar arithmetic, filling masked entries, compression,
valid-value counting, mixed indexing, and axis-wise gather/update that keeps
values and masks aligned. Sum, product, min, max, mean, variance, standard
deviation, median, linearly interpolated quantiles and percentiles, and
masked-aware argmin/argmax work globally, along one axis, or along multiple
axes. Axis and multi-axis reductions support `keepdims`; negative axes are
normalized and repeated or out-of-range axes are rejected. Arg reductions
return the first valid matching index and mark all-masked slices in the output
mask. Axis reductions mark empty slices in the output mask;
global reductions expose an `is_masked` flag when every value is missing.
Cumulative sum and product work over flattened logical order or along one axis;
masked positions stay masked and contribute their operation's identity to the
running accumulator. Explicit output accumulator types are available for both
operations. See the [masked data tutorial](./TUTORIAL_MASKED_DATA.md).

**Remaining:** additional masked statistics, NumPy quantile interpolation
methods beyond the current linear method, and interoperability with autograd
and neural-network operations. NaN-aware reductions are separate and do not
automatically treat NaN as masked data.

### Structured and record arrays

**Current:** not supported as a general tensor feature.

**Remaining:** decide whether this belongs in VTL or a companion
table/dataframe package.

### Performance and devices

**Current:** pure-V CPU and optional CBLAS paths, f32 GEMM, and optional GPU
backends. Contiguous `f64` p=2 vector norms use VSL's backend-dispatching
`dnrm2`, so CBLAS builds can call the configured implementation.
OpenCL currently covers tensor transfer and a small `add_scalar` smoke path;
broader compute and neural-network dispatch are tracked in
[#750](https://github.com/vlang/vtl/issues/750).

**Remaining:** publish reproducible per-backend benchmarks and optimize while
preserving numerical behavior.

### Learning and examples

**Current:** autograd, neural networks, optimizers, datasets, and tutorials.

**Remaining:** add end-to-end examples for classical ML, transforms, batching,
checkpointing, and deployment.

The NumPy `.npy` format stores dtype, shape, and memory order alongside binary
data; `.npz` is a ZIP container of `.npy` files. VTL's `.npz` reader supports
named members, compressed and uncompressed archives, and reads member bytes
without extracting paths. Its `write` function accepts tensors of one element
type; `write_arrays` accepts mixed types wrapped through `array` and encodes
each member while writing. The `.npy` reader supports v1, v2, and v3 headers, both
byte orders, and Fortran-order arrays. Unsupported object, string, and
structured dtypes are rejected rather than converted silently; complex128
`.npy` and `.npz` members are supported. Interop tests use a compressed archive
generated by NumPy. Memory mapping remains future work.

`allclose` reduces comparisons without allocating an intermediate boolean
tensor. It uses direct access when both operands are contiguous and have the
same shape, and stops at the first mismatch; the broadcast path preserves
NumPy-style shape semantics.

`stats.median` accepts unsorted input, uses quickselect-backed linear
interpolation, and returns fractional `f64` results for integer tensors.
The `nanmedian` variants provide the matching NaN-ignoring scalar and axis
reductions. Percentile APIs now cover scalar and per-axis operations, with
NaN-ignoring variants, multiple requested values, and
keep-dimension/squeezed output shapes.

## Arraymancer comparison

Track these capabilities in addition to the NumPy table:

- Multidimensional tensor math, slicing, broadcasting, reshape, concatenation,
  permutation, and matrix algebra.
- CSV/NPY/HDF5 data interchange and CPU/GPU backends. VTL supports numeric
  CSV tensors, typed `.npy` arrays, and named `.npz` members with mixed
  supported dtypes; optional HDF5 interoperability is tracked in
  [#779](https://github.com/vlang/vtl/issues/779).
- Statistics, covariance, eigen/least-squares operations, PCA, and K-means.
- Neural-network layers and recurrent models, with training examples.

VTL already has more developed autograd, neural-network, optimizer, dataset,
and multi-backend training infrastructure than a basic tensor library. The
remaining comparison must verify operational feature behavior and performance
against current upstream documentation; feature names alone are not evidence.

## Acceptance rule

For each capability, record its public API, correctness tests, edge cases,
documentation/tutorial, example, and benchmark (when performance matters).
Keep unsupported or partially supported features explicitly marked. Do not
describe VTL as more complete than NumPy or Arraymancer until the applicable
rows are implemented and independently verified.

[i725]: https://github.com/vlang/vtl/issues/725
[i726]: https://github.com/vlang/vtl/issues/726
[i779]: https://github.com/vlang/vtl/issues/779
