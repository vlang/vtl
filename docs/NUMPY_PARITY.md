# NumPy and Arraymancer Feature Parity

VTL's long-term goal is to provide a more complete and more performant tensor
and machine-learning toolkit than Arraymancer, then grow beyond NumPy for the
array operations that make sense in V. This is a capability roadmap, not a
claim of current parity. Every checked item must be backed by public API,
tests, documentation, and examples where users need them.

## Current strengths

- Generic N-dimensional tensors, slicing, broadcasting, reshape, transpose,
  axis swapping, and view-preserving `moveaxis`/`rollaxis`.
- Elementwise math, reductions, mapping, stacking, splitting, random values,
  statistics, and linear algebra through VSL.
- Reverse-mode autograd, neural-network layers and losses, optimizers,
  datasets, and CPU/CUDA/Vulkan/OpenCL paths.
- Tutorials and runnable examples for tensors, linear algebra, autograd,
  training, and GPU backends.

## NumPy domains

### Array creation

**Current:** zeros, ones, full, eye, range, sequence, arrays, and Vandermonde
matrices; `arange` with start/stop/step; endpoint-aware `linspace`; configurable
`logspace`; 2-D `meshgrid` and N-D `meshgrid_n` with `xy` and `ij` indexing;
dense stacked coordinates through `indices` and broadcastable per-axis sparse
coordinates through `indices_sparse`.

**Remaining:** validate edge cases and make dtype/device options consistent.

### Shape and manipulation

**Current:** reshape, transpose, squeeze/expand, move/roll axes, N-D
`diagonal` views over two selected axes with NumPy-style offset and output
shape, concatenate/stack/split, `ravel`, copying `flatten`, `flip`, `repeat`,
`repeat_axis`, block `tile`, `rot90`, constant/edge/wrap/reflect/symmetric
padding, and stable axis-wise `sort`/`argsort`.
Top-level `tril` and `triu` copy N-D inputs and apply NumPy's diagonal offset
to each trailing matrix.

**Current:** `broadcast_to`, `broadcast2`, `broadcast3`, and `broadcast_n`
create zero-copy views, validate incompatible shapes, and follow NumPy's
zero-dimension rules. Tests cover empty inputs, rank mismatches, and invalid
target dimensions.

**Remaining:** audit copy-versus-view behavior across other manipulation APIs.
VTL diagonal views share writable storage; NumPy exposes its diagonal views as
read-only by default.

### Indexing and set operations

**Current:** integer indexing and slices; broadcasted coordinate-array
`advanced_index`; mixed `mixed_index` selectors for coordinate arrays, scalar
indices, slices, ellipsis, and new axes; sorted `unique` and its counts,
inverse, and first-index variants;
`unique_axis`; `digitize`; ascending and descending `searchsorted`; global and
axis `count_nonzero`; `argwhere` and per-axis `nonzero`; `take`, `take_nd`,
`take_flat`; `take_along_axis` with broadcasting outside the selected axis;
`put_along_axis` and `scatter_add`; lookup and DataLoader gathers; broadcastable
`masked_select` and `masked_fill`.

`Variable.take_along_axis` propagates gradients by scattering them to selected
source positions, including accumulation for repeated indices and reduction of
broadcast source dimensions.
`Variable.scatter_add` returns an updated copy and propagates gradients to its
source and update values, including repeated destination indices.
`Variable.put_along_axis` follows last-write-wins semantics and routes gradients
only to the final update at each destination.
`Variable.slice` and `Variable.slice_hilo` propagate gradients through integer
indices, range views, and positive steps.
`Variable.sum` and `Variable.mean` reduce all elements to a one-element tensor
and propagate gradients to every input element. Their `*_along_axis` variants
support one-axis sum and mean, with either squeezed or retained dimensions.

**Remaining:** document all bounds semantics.

Axis-wise `argmax` and `argmin` retain a length-one axis for compatibility;
`argmax_axis_squeeze` and `argmin_axis_squeeze` provide NumPy's default
axis-removed output shape. VTL represents scalar-shaped reductions as shape
`[1]`.

### Math and ufuncs

**Current:** broad elementwise math and broadcasting; `clip` and fused
`clip_tensor`; broadcast-aware `where`; elementwise `is_nan`, `is_inf`, and
`is_finite`; `isclose`, `allclose`, `heaviside`, `sign`, `diff`, and numerical
gradients along selected axes; elementwise logical AND/OR/XOR/NOT; integer
bitwise AND/OR/XOR/invert and scalar left/right shifts. Gradients support
uniform spacing or monotonic non-uniform coordinates and first/second-order
boundary differences.

**Remaining:** audit unary/binary function families and define dtype promotion.

### Reductions and statistics

**Current:** single-axis and multi-axis sum/product, NaN-aware sum/product/
min/max, and logical all/any with `keepdims`; weighted average globally and
along an axis; scalar population/sample variance and
standard deviation; single-axis and multi-axis mean/variance/std with explicit
`keepdims` and NaN-aware forms; axis-wise `trapezoid`; linear quantile/percentile and
multi-quantile APIs, including NaN-aware global and axis variants; axis arg
reductions and squeezed min/max/argmin/argmax results; multi-axis min/max;
histograms with
automatic and custom bins, weighted density, and multiple bin-selection rules;
integer/weighted `bincount`.

**Remaining:** add `keepdims` options consistently across remaining reductions,
accumulator dtype controls, and broader reduction families.

Logical `all_axis`/`any_axis` and multi-axis `all_axes`/`any_axes` reductions
return `bool` tensors; empty reduced dimensions return the logical identities
(true and false respectively). Empty axis lists perform elementwise truth
conversion, matching NumPy's `axis=()` behavior.

### Linear algebra

**Current:** NumPy-style vector/matrix `matmul` promotion and batched
broadcasting; N-D Kronecker products with NumPy rank-promotion semantics;
offset diagonal construction and extraction; solve,
QR/LU/Cholesky, pseudoinverse, trace, matrix norms,
flattened and axis-wise p-norms, and covariance/correlation matrices.

`la.matrix_norm` computes NumPy's eight standard matrix norms for each trailing
matrix in an N-D tensor, with optional retained matrix dimensions.
`la.trace_axes` sums offset diagonals across selected axes and preserves the
remaining dimensions as a batch.
`la.svdvals` returns descending singular values for batched real matrices.
`la.slogdet` returns determinant signs and log absolute determinants for batches,
including singular matrices and values whose determinants would overflow.
`la.matrix_power` uses exponentiation by squaring for batched integer powers and
partial-pivot inversion for negative exponents.
`la.cond` computes spectral and induced-norm condition numbers for stacked
square matrices, with singular spectral cases returning infinity or zero for
orders `2` and `-2` respectively.
`la.det` and `la.inv` operate on stacks of square matrices; inverse results
are checked against their input layout and singular inputs report errors.

**Remaining:** expand eigen/SVD options, specify singular/non-finite behavior,
and benchmark realistic shapes.

### FFT

**Current:** 1-D, selected-axis, and N-D complex transforms for f32/f64; real
`rfft`/`rfftn` and inverses; reusable real plans; `fftfreq`/`rfftfreq` and
`fftshift`/`ifftshift`; NumPy's `backward`/`forward`/`ortho` normalization.
Dedicated `_f32` real APIs preserve f32 output; generic `rfft[f32]` retains its
f64 complex return type.

**Remaining:** benchmark large/strided arrays and complete the dtype-promotion
audit.

### Random

**Current:** global uniform, normal, Bernoulli, binomial, geometric,
exponential, Poisson, Weibull, lognormal, gamma, beta, Dirichlet, chi-square,
Student's t, and F tensors; `random_seed`; independent seeded `RandomGenerator`
streams for f64 uniform, normal, lognormal, gamma, beta, Dirichlet, exponential,
Poisson, Weibull, chi-square, Student's t, F, and boolean Bernoulli; integer
binomial/geometric/multinomial/Poisson; uniform and weighted population choice;
and integer or first-axis tensor permutations.

**Remaining:** add distributions/sampling APIs and define reproducibility across
runtime versions.

### Input and output

**Current:** model serialization and dataset loaders; numeric CSV tensors;
typed `.npy` and named `.npz` I/O with mixed supported dtypes. `.npy` accepts
v1/v2/v3 headers, both byte orders, Fortran order, bool, f32/f64, and signed or
unsigned integer widths when the requested V type matches.

**Remaining:** support more text formats and dtypes, add independent NumPy
fixtures, and continue rejecting unsupported object/structured values.

### Data types

**Current:** V generic element types, `dtype()` introspection, and
`promote_types` for supported array dtypes. `math.complex.Complex` is supported
as `complex128` tensor storage with elementwise add, subtract, multiply, and
divide. Promotion with a complex operand produces `complex128`. Explicit
`as_*` conversions cover real numeric and boolean types, preserve shape and
logical order, and truncate floating-point values when converting to integers.

**Remaining:** connect promotion to mixed-dtype arithmetic, add weak scalar
promotion and string conversion, extend complex support to mathematical
functions, reductions, linear algebra, random generation, and `.npy`/`.npz` I/O,
add complex32, and define integer overflow and structured-data limits.

### Masked and missing data

**Current:** `MaskedArray[T]` pairs values with a broadcastable boolean mask
(`true` means missing). It supports broadcasted arithmetic and comparisons
with mask union, scalar arithmetic, filling masked entries, compression,
valid-value counting, mixed indexing, and axis-wise gather/update that keeps
values and masks aligned. Sum, product, min, max, mean, variance, and standard
deviation work globally and across one or multiple axes. Axis reductions mark
empty slices in the output mask; global reductions expose an `is_masked` flag
when every value is missing. See
[the masked data tutorial](./TUTORIAL_MASKED_DATA.md).

**Remaining:** additional reductions and statistics, and interoperability with
autograd and neural-network operations. NaN-aware reductions are separate and
do not automatically treat NaN as masked data.

### Structured and record arrays

**Current:** not supported as a general tensor feature.

**Remaining:** decide whether this belongs in VTL or a companion
table/dataframe package.

### Performance and devices

**Current:** pure-V CPU and optional CBLAS paths, f32 GEMM, and optional GPU
backends. Contiguous `f64` p=2 vector norms use VSL's backend-dispatching
`dnrm2`, so CBLAS builds can call the configured implementation.

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
byte orders, and Fortran-order arrays. Unsupported object, string, structured,
and complex dtypes are rejected rather than converted silently. Interop tests
use a compressed archive generated by NumPy. Memory mapping remains future
work.

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
  supported dtypes; HDF5 is not yet provided by VTL itself.
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
