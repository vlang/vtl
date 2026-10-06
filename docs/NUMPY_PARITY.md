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

| Domain | Current VTL coverage | Work needed before calling it complete |
|---|---|---|
| Array creation | zeros/ones/full/eye/range/sequence, `arange` with start/stop/step, endpoint-aware `linspace`, configurable `logspace`, from arrays, two-vector 2-D `meshgrid`, N-D `meshgrid_n` with `xy` and `ij` indexing | Validate edge cases and consistent dtype/device options |
| Shape and manipulation | reshape, transpose, squeeze/expand, move/roll axes, concatenate/stack/split, `ravel`, copying `flatten`, `flip`, `repeat`, `repeat_axis`, block `tile`, `rot90`, stable `sort` and `argsort` over selected axes | Add broadcasting helper tests and a broader copy-vs-view audit |
| Indexing and set operations | Integer indexing, slices, sorted `unique`, `unique_counts`, flattened `unique_inverse` and `unique_first_indices`, lexicographic `unique_axis` metadata, monotonic `digitize`, ascending and descending binary `searchsorted`, global and axis `count_nonzero` with `keepdims`, `argwhere`, per-axis `nonzero` coordinate arrays, `take`/`take_nd`/`take_flat`/`take_along_axis` gathers, `put_along_axis`, additive `scatter_add`, lookup helpers, DataLoader gathers, broadcastable boolean `masked_select`/`masked_fill` | Add general fancy indexing, autograd support for indexed updates, and document all bounds semantics |
| Math and ufuncs | Broad elementwise math and broadcasting; scalar-bound `clip` plus fused `clip_tensor` with broadcastable tensor bounds, broadcast-aware `where`, `isclose`, `allclose`, `heaviside`, `sign`, and discrete differences with `diff` | Audit the full unary/binary function families and dtype promotion |
| Reductions | `sum_along_axis` and `product_along_axis` return tensors and support `keepdims`; `sum_along_axes` and `product_along_axes` reduce multiple axes at once, including negative axes and empty-axis copies; weighted average globally and along an axis; scalar population/sample variance and standard deviation; mean/variance/std along an axis; `trapezoid` integration along an axis; linearly interpolated scalar/vector `quantile_linear`/`percentile_linear`, keep-dimension and NumPy-shaped squeezed `quantile_axis` variants, and `quantiles_axis`; NaN-aware mean/variance/std, sum/product/min/max, and quantiles globally and along an axis, including squeezed and keep-dimension single quantiles and `nanquantiles_axis`; axis arg reductions; evenly spaced and custom-edge histograms, weighted density, and automatic/Sturges/Doane/square-root/Rice/Scott/Freedman-Diaconis/Stone bin rules; integer and weighted `bincount` | Broader `keepdims` support across reduction families, accumulator dtype controls, and broader reduction families |
| Linear algebra | NumPy-style vector and matrix `matmul` promotion/batched broadcasting, solve, QR/LU/Cholesky, pseudoinverse, trace, matrix norms, flattened vector p-norms, single- and tuple-axis p-norm reductions, and NumPy-oriented covariance/correlation matrices | Add eigen/SVD option coverage, specify singular and non-finite behavior, and benchmark realistic shapes |
| FFT | 1D, selected-axis, and N-D complex transforms, real `rfft`/`rfftn` and inverse transforms along one or all axes, reusable 1D real plans, `fftfreq`/`rfftfreq`, and `fftshift`/`ifftshift` across all axes or one axis | Add explicit normalization modes and complex f32 transforms; benchmark large and strided arrays |
| Random | Uniform range, normal, Bernoulli, binomial, geometric, gamma, beta, and exponential tensors; seeded `choice` sampling with/without replacement; reproducible global `random_seed`; independent seeded `RandomGenerator` streams for f64 uniform/normal/gamma/beta, bool Bernoulli, integer geometric, and population choice | Add broader distributions and sampling APIs; define cross-version reproducibility guarantees |
| Input/output | Model serialization, dataset loaders, numeric CSV tensor read/write, typed `.npy` tensor read/write, and `.npz` named-member read/write including mixed supported dtypes; `.npy` reader accepts v1/v2/v3 headers, both byte orders, and Fortran-order arrays. Supported `.npy` values include bool, f32/f64, signed integer widths, and unsigned integer widths when the requested V type matches | Add broader text format support, broader NumPy dtype support, and more independent fixtures generated by NumPy; keep unsupported object/structured data rejected |
| Data types | V generic element types | Define promotion/casting rules, complex types, booleans, integer overflow, and structured data limits |
| Masked and missing data | Not established as a general tensor feature | Define a mask representation and NaN/missing-value reduction behavior |
| Structured/record arrays | Not supported as a general tensor feature | Decide whether this belongs in VTL or a companion table/dataframe package |
| Performance and devices | Pure-V CPU and optional CBLAS CPU paths; `f32` matmul uses single-precision GEMM with CBLAS flags, alongside optional GPU backends | Publish reproducible benchmarks for each backend; optimize without changing numerical semantics |
| Learning and examples | Autograd, NN, optimizers, datasets, tutorials | Add end-to-end examples for classical ML, transforms, batching, checkpointing, and deployment |

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
