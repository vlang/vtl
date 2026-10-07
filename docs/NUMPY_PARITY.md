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

**Current:** zeros, ones, full, eye, range, sequence, and arrays; `arange` with
start/stop/step; endpoint-aware `linspace`; configurable `logspace`; 2-D
`meshgrid` and N-D `meshgrid_n` with `xy` and `ij` indexing; dense stacked
coordinate arrays through `indices`.

**Remaining:** validate edge cases and make dtype/device options consistent.

### Shape and manipulation

**Current:** reshape, transpose, squeeze/expand, move/roll axes,
concatenate/stack/split, `ravel`, copying `flatten`, `flip`, `repeat`,
`repeat_axis`, block `tile`, `rot90`, and stable axis-wise `sort`/`argsort`.

**Remaining:** test broadcasting helpers and audit copy-versus-view behavior.

### Indexing and set operations

**Current:** integer indexing and slices; broadcasted coordinate-array
`advanced_index`; sorted `unique` and its counts/inverse/first-index variants;
`unique_axis`; `digitize`; ascending and descending `searchsorted`; global and
axis `count_nonzero`; `argwhere` and per-axis `nonzero`; `take`, `take_nd`,
`take_flat`; `take_along_axis` with broadcasting outside the selected axis;
`put_along_axis` and `scatter_add`; lookup and DataLoader gathers; broadcastable
`masked_select` and `masked_fill`.

**Remaining:** mix coordinate arrays with slices, add autograd support for
indexed updates, and document all bounds semantics.

### Math and ufuncs

**Current:** broad elementwise math and broadcasting; `clip` and fused
`clip_tensor`; broadcast-aware `where`; `isclose`, `allclose`, `heaviside`,
`sign`, `diff`, and numerical gradients along selected axes. Gradients support
uniform spacing or monotonic non-uniform coordinates and first/second-order
boundary differences.

**Remaining:** audit unary/binary function families and define dtype promotion.

### Reductions and statistics

**Current:** single-axis and multi-axis sum/product with `keepdims`; weighted
average globally and along an axis; scalar population/sample variance and
standard deviation; axis mean/variance/std with explicit `keepdims` and
NaN-aware forms; axis-wise `trapezoid`; linear quantile/percentile and
multi-quantile APIs, including NaN-aware global and axis variants; axis arg
reductions; histograms with automatic and custom bins, weighted density, and
multiple bin-selection rules; integer/weighted
`bincount`.

**Remaining:** extend `keepdims` to the remaining reduction families and add
accumulator dtype controls and broader reductions.

### Linear algebra

**Current:** NumPy-style vector/matrix `matmul` promotion and batched
broadcasting; solve, QR/LU/Cholesky, pseudoinverse, trace, matrix norms,
flattened and axis-wise p-norms, and covariance/correlation matrices.

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

**Current:** uniform, normal, Bernoulli, binomial, geometric, gamma, beta, and
exponential tensors; seeded `choice`; global `random_seed`; independent seeded
`RandomGenerator` streams for f64 uniform/normal/gamma/beta, boolean Bernoulli,
integer geometric, and population choice.

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
`promote_types` for supported array dtypes. Explicit `as_*` conversions cover
numeric and boolean types, preserve shape and logical order, and truncate
floating-point values when converting to integers.

**Remaining:** connect promotion to mixed-dtype arithmetic, add weak scalar
promotion and string conversion, support complex types, and define integer
overflow and structured-data limits.

### Masked and missing data

**Current:** no general tensor-level masked-data representation.

**Remaining:** define masks and NaN/missing-value reduction behavior.

### Structured and record arrays

**Current:** not supported as a general tensor feature.

**Remaining:** decide whether this belongs in VTL or a companion
table/dataframe package.

### Performance and devices

**Current:** pure-V CPU and optional CBLAS paths, f32 GEMM, and optional GPU
backends.

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
