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
| Array I/O formats | [#726][i726], [#779][i779] |
| Masked autograd and structured data | [#727](https://github.com/vlang/vtl/issues/727), [#728](https://github.com/vlang/vtl/issues/728) |
| Backend benchmarks | [#729](https://github.com/vlang/vtl/issues/729) |
| OpenCL compute and neural-network dispatch | [#750](https://github.com/vlang/vtl/issues/750) |
| Sparse tensor operations | [#746](https://github.com/vlang/vtl/issues/746) |
| Non-numeric dtypes and array interoperability | [#747](https://github.com/vlang/vtl/issues/747), [#748](https://github.com/vlang/vtl/issues/748) |
| NumPy polynomial families and fitting | [#773](https://github.com/vlang/vtl/issues/773) (power-series subset implemented) |

For neural-network and framework-level gaps relative to PyTorch, see the
[PyTorch parity tracker](./PYTORCH_PARITY.md).

## Current strengths

- Generic N-dimensional tensors, slicing, broadcasting, reshape, transpose,
  axis swapping, and view-preserving `moveaxis`/`rollaxis`.
- Elementwise math, reductions, mapping, stacking, splitting, random values,
  statistics, and linear algebra through VSL.
- `ufunc_unary`, `ufunc_unary_into`, `ufunc_binary`, and `ufunc_binary_into`
  define allocation-aware unary and binary execution paths with caller-owned
  `out` tensors and exact-shape `where` masks; masked entries preserve their
  preallocated output values. `ufunc_reduce` and `ufunc_accumulate` currently
  operate in logical flat row-major order and accept an explicit identity.
  `ufunc_binary` broadcasts operands; `ufunc_unary_into` and
  `ufunc_binary_into` reuse caller-owned outputs and support broadcastable
  `where` masks without allowing the mask to expand the result shape.
  `ufunc_outer`, flattened `ufunc_reduceat`, axis-aware
  `ufunc_reduce_axis`/`ufunc_accumulate_axis`, and `ufunc_at` are also
  available. `ufunc_at` performs sequential indexed mutation with repeated-index
  accumulation and prevalidates bounds. The helpers require one static element
  type and explicit reduction identities; NumPy's casting modes and optional
  retained dimensions are not exposed.
- A power-series `Polynomial` API with ascending coefficients, Horner
  evaluation, tensor evaluation, arithmetic, derivative/integral, real
  linear and quadratic roots, and small least-squares fitting. Orthogonal
  families and higher-degree robust root solving remain open. See the runnable
  [polynomial fitting example](../examples/polynomial_fit/main.v).
- Reverse-mode autograd, neural-network layers and losses, optimizers,
  datasets, CPU support, and selected CUDA/Vulkan operations. OpenCL/VCL
  currently covers tensor transfer and an `add_scalar` smoke example; VTL
  OpenCL compute and neural-network dispatch remain incomplete (see the
  [backend roadmap](../ROADMAP.md#phase-5--opencl-backend)).
- Tutorials and runnable examples for tensors, linear algebra, autograd,
  training, and GPU backends.

## NumPy domains

### Ufunc execution protocol

| NumPy protocol | VTL status |
|---|---|
| Unary/binary elementwise execution | `ufunc_unary`, `ufunc_binary`; binary operands broadcast |
| Preallocated `out` | `ufunc_unary_into`, `ufunc_binary_into`; result shape must be supplied exactly |
| `where` | Supported for `*_into`; masks broadcast to, but cannot enlarge, the result shape; false entries preserve `out` |
| `reduce` / `accumulate` | `ufunc_reduce`, `ufunc_accumulate` for flattened logical row-major order; axis variants are `ufunc_reduce_axis` and `ufunc_accumulate_axis` |
| `reduceat` / `outer` / `at` | `ufunc_reduceat` is flattened, `ufunc_outer` uses flattened operands, `ufunc_at` mutates flattened indices sequentially |
| Casting modes | Not implemented; input/output types are the same generic type |
| NumPy error-state controls and `where`-masked allocation semantics | Not implemented |

### Array creation

**Current:** zeros, ones, full, eye, range, sequence, arrays, and Vandermonde
matrices; `arange` with start/stop/step; endpoint-aware `linspace`; configurable
`logspace`; 2-D `meshgrid` and N-D `meshgrid_n` with `xy` and `ij` indexing;
dense stacked coordinates through `indices` and broadcastable per-axis sparse
coordinates through `indices_sparse`. VTL also provides `one_hot` for ML labels,
appending a class dimension and validating class bounds. Element dtype is
selected through each constructor's generic type argument. Constructors that
accept `TensorData`, `LinspaceData`, or `LogspaceData` accept a memory format;
creation tests verify column-major layout for the scalar-fill and sequence
families.
Flat `from_array` values follow the requested physical memory order; nested
`from_2d` rows remain logical row-major values and are rearranged for
column-major storage.

**Remaining:** constructor APIs do not expose a common device option. `meshgrid`,
`meshgrid_n`, `indices`, `indices_sparse`, and `one_hot` currently use their
fixed row-major layout. Document and validate those differences, edge cases,
and device support as the storage/device model evolves.

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
`transpose`, `moveaxis`, `squeeze`, `unsqueeze`, and `expand_dims` preserve
shared storage as views, including for transposed input.
Basic `slice`, `slice_hilo`, and `mixed_index` selections are writable,
storage-sharing views; advanced coordinate and boolean indexing return copies.
Mutation tests cover slice views on contiguous and transposed inputs.
`split` returns storage-sharing slices, including when splitting a transposed
input. `diagonal_readonly` provides a storage-sharing diagonal view that
exposes only read operations and returns copied shape/value arrays.
`concatenate`, `stack`, `repeat`, `repeat_axis`, `tile`, `flip`, `rot90`,
`pad`, and `flatten` return independent copies; mutation tests cover transposed
inputs for these allocation-based operations.
`ravel` shares storage for row-major contiguous input and copies other layouts;
`flatten` always copies. Mutation tests verify these contracts for contiguous
and transposed inputs.
Top-level `tril` and `triu` copy N-D inputs and apply NumPy's diagonal offset
to each trailing matrix.

**Current:** `broadcast_to`, `broadcast2`, `broadcast3`, and `broadcast_n`
create writable zero-copy views, validate incompatible shapes, and follow
NumPy's zero-dimension rules. Writes through broadcasted coordinates update
their source, and zero-stride dimensions alias the same source element. Tests
cover empty inputs, invalid shapes, and mutation through contiguous and
transposed sources.

**Known mutability difference:** NumPy exposes broadcast and diagonal views as
read-only by default. VTL retains writable `broadcast_to` and legacy `diagonal`
for compatibility; `diagonal_readonly` provides an explicit read-only
alternative for diagonal access.

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
`put_along_axis` and `scatter_add` accept negative indices, raise on out-of-range
indices, and validate all indices before mutating; repeated destinations use
last-write-wins and accumulation semantics, respectively.
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

The bounds and duplicate-update contracts are intentionally API-specific:

| API | Bounds behavior | Duplicate behavior |
| --- | --- | --- |
| `Tensor.get` / `set` / `offset_index` | Negative coordinates count from the end; invalid rank or coordinates panic. `get_checked` / `set_checked` return errors. No mode parameter. | One coordinate addresses one element. |
| `advanced_index` / `mixed_index` / `boolean_index` | Raise-only on invalid coordinates or mask shapes, matching NumPy indexing. Negative in-range coordinates count from the end. | Repeated coordinates repeat selected output values; these APIs do not write. |
| `take` / `take_nd` / `take_flat` | `TakeMode.raise`, `.wrap`, and `.clip`; default is `.raise`. | Repeated indices repeat gathered values. |
| `take_along_axis` | Raise-only with negative in-range indices counting from the end, matching NumPy. | Repeated indices repeat gathered values. |
| `put` / `put_with_mode` | `PutMode.raise`, `.wrap`, and `.clip`; default is `.raise`. All indices validate before writes. | Repeated destinations use row-major last-write-wins; shorter update tensors repeat. |
| `put_along_axis` | Raise-only with negative in-range indices counting from the end, matching NumPy. All indices validate before writes. | Repeated destinations use row-major last-write-wins. |
| `scatter_add` | Raise-only with negative in-range indices counting from the end. All indices validate before writes. | Repeated destinations accumulate in row-major order. |
| `putmask` | The mask must match the target shape; no integer bounds mode applies. | Values cycle by destination flat position. |
| `ravel_multi_index` | Supports global or per-axis `raise`, `wrap`, and `clip` modes. | Repeated coordinates produce repeated flat indices. |
| `unravel_index` | Raise-only when flat indices are outside the selected order's valid range. | Repeated flat indices repeat their coordinate values. |

`Variable.take`, `Variable.take_nd`, `Variable.take_flat`,
`Variable.take_along_axis`, `Variable.mixed_index`, `Variable.boolean_index`, and
`Variable.masked_select` propagate gradients by scattering them to selected
source positions, including accumulation for repeated indices and reduction of
broadcast source dimensions. `Variable.mixed_index` snapshots the resolved
logical source positions during the forward pass, so later index mutations do
not affect backward. `Variable.masked_fill` zeros gradients at replaced input
positions.
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

`take_along_axis`, coordinate/boolean indexing, axis-wise replacement, and
`scatter_add` intentionally remain raise-only; NumPy does not define alternate
bounds modes for these indexing operations. Tensor indexed writes remain
in-place and non-differentiable; use the `Variable` APIs when gradients are
required.

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

**Current:** broad elementwise math and broadcasting, including dtype-preserving
`square`; same-dtype tensor `+`,
`-`, `*`, and `/` operators; `clip` and fused
`clip_tensor`; broadcast-aware `where`; elementwise `is_nan`, `is_inf`, and
`is_finite`; `isclose`, `allclose`, `heaviside`, `sign`, `diff`, and numerical
gradients along selected axes; elementwise logical AND/OR/XOR/NOT; integer
bitwise AND/OR/XOR/invert and scalar left/right shifts. Gradients support
uniform spacing or monotonic non-uniform coordinates and first/second-order
boundary differences. `sign` normalizes complex values to their phase, matching
NumPy 2.0+, returning zero for zero, `nan+nanj` if either component is NaN,
the unit axis for a single infinite component, and `nan+nanj` for two infinite
components.
`Tensor.reciprocal()` computes `1/x` for f32/f64
tensors, preserves shape and dtype, and follows IEEE signs for zeros and
infinities. Integer and complex inputs are intentionally unsupported.
`Tensor.sinc()` computes `sin(pi*x)/(pi*x)`, defines
both signed zeros as `1`, and propagates NaN for NaN or infinite inputs. It
accepts only f32/f64 tensors and preserves their dtype and shape; unlike NumPy,
it does not promote integer inputs. Promoted math functions include
`atan2_promoted`, `sqrt_promoted`, `cbrt_promoted`, `exp_promoted`,
`exp2_promoted`, `expm1_promoted`, `log_promoted`,
`log2_promoted`, `log10_promoted`, `log1p_promoted`, `sin_promoted`,
`cos_promoted`, and `tan_promoted`, plus the inverse-trigonometric and direct
and inverse hyperbolic functions. They provide NumPy-style floating-point
outputs for real numeric or boolean inputs, returning f64 for f64 or
32-bit-and-wider integer inputs, and f32 otherwise.
The same dtype rules apply to the broadcasted binary `copysign_promoted`,
`hypot_promoted`, `logaddexp_promoted`, `logaddexp2_promoted`, and
`nextafter_promoted`.
`rint_promoted` follows the same floating output rules and uses ties-to-even;
`floor_promoted`, `ceil_promoted`, and `trunc_promoted` preserve the input
dtype, including exact large integer values.
The explicit `vtl.array_api` namespace also exposes whole-array and
axis-aware `std`, `var`, `median`, and linear-interpolated `quantile`
reductions with documented `ddof` handling and a `keepdims` option.
NumPy float16 results for booleans and 8-bit integers are widened because VTL
has no float16. The legacy same-dtype `Tensor.atan2()`, `Tensor.sqrt()`, and
`Tensor.cbrt()` methods preserve their generic input dtype; use the promoted
functions for integer or mixed-dtype results. Complex and string inputs are
unsupported.
Same-dtype `floor_divide` and `floor_divide_scalar`
round signed quotients toward negative infinity, preserve input dtype, return
zero for integer zero divisors, and follow IEEE results for floating-point
zero divisors. Mixed-dtype promotion remains tracked in
[#717](https://github.com/vlang/vtl/issues/717). `add_promoted`, `subtract_promoted`, and
`multiply_promoted` accept mixed real and complex numeric dtypes, broadcast
their inputs, and validate an explicitly selected output dtype against
`promote_types`. `divide_promoted` performs NumPy-style true division for mixed
real numeric, complex, and boolean tensors, broadcasts inputs, and returns
`complex128` when either input is complex; real inputs return `float32` only
when the promoted input dtype is `float32`, and `float64` otherwise.
`remainder_promoted` performs NumPy-style remainder with mixed integer and
floating-point dtypes and broadcasting. Nonzero results follow the divisor's
sign; signed zero follows the divisor; integer zero divisors produce zero and
floating-point zero divisors produce NaN; complex inputs are rejected.
`pow_promoted` broadcasts mixed real numeric tensors and validates its explicit
output dtype against the operation's NumPy result dtype: numeric pairs use
`promote_types`, while boolean pairs return `int8`. Integer powers retain
exact-width wrapping arithmetic and reject negative integer exponents. Floating
results preserve IEEE special values and signed zero. Complex and string inputs
are unsupported; 64-bit integer operands are rejected when promotion requires a
floating result to avoid silent precision loss.
`float_power_promoted` provides the NumPy `float_power` contract for real
numeric and boolean tensor pairs: it always returns f64, including for f32
inputs, and allows negative integer exponents. Inputs are converted to f64, so
large integers may lose precision just as they do with NumPy.
`fmod_promoted` and its weak-scalar variants preserve the sign-of-dividend
remainder rule across promoted real numeric dtypes and broadcasting. Integer
operations retain exact-width arithmetic; zero divisors and signed minimum
modulo `-1` produce zero. Floating-point operations use `math.fmod`, including
IEEE NaN results and signed-zero preservation. Boolean and complex inputs are
unsupported.
`maximum_promoted`, `minimum_promoted`, `fmax_promoted`, and `fmin_promoted`
broadcast mixed numeric, boolean, or complex tensor pairs using `promote_types`.
The first pair propagates NaNs; the latter pair ignores one NaN operand. Integer
comparisons retain exactness when promotion produces an integer dtype, and
floating-point tie handling matches NumPy's signed-zero rules. Weak-scalar
extrema remain part of #717.
`ldexp_promoted` broadcasts real numeric or boolean mantissas with integer or
boolean exponents. Floating mantissas preserve f32/f64; integer and boolean
mantissas produce f64, matching NumPy's available result loops.
`frexp_promoted` accepts real numeric and boolean inputs, returning f32
mantissas for f32 inputs and f64 mantissas otherwise, alongside integer
exponents.
`modf_promoted` accepts the same inputs and returns fractional and integral
parts using the promoted floating dtype.
Floating-point autograd is available
through `Variable.remainder` for f32/f64: the divisor gradient uses
`-floor(a / b) * grad`, including at exact multiples, and broadcast gradients
are reduced to each input shape. Non-finite values follow the forward
operation's IEEE results and the same derivative formula. The rule follows
[PyTorch's remainder derivative](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml#L1311-L1317).
See
[issue #798](https://github.com/vlang/vtl/issues/798), the
[implementation](../math_promoted.v), [forward tests](../tests/core/math_promoted_remainder_test.v),
[gradient tests](../autograd_tests/gates_remainder_test.v), and
[dtype conversion tutorial](./TUTORIAL_FIRST_STEPS.md#numeric-dtype-conversion).

**Remaining ufunc gaps:** complex `pow` remains unsupported, and unary
`abs`/`sign` plus multi-output `frexp`/`modf` still need broader differential
coverage across supported dtypes and special values. Promoted `float_power`
now covers real numeric and boolean tensor pairs, broadcasting, empty and
scalar-shaped inputs, negative integer exponents, and IEEE edge cases.
Promoted mixed-input `ldexp` covers real or boolean mantissas and integer or
boolean exponents; promoted extrema cover tensor pairs and weak scalars.
`erf`/`erfc`, `gamma`, and `log_gamma` are useful VTL math APIs beyond NumPy
core ufunc parity. The shared `out`/`where` helper contract is documented in the
[ufunc execution capability table](#ufunc-execution-protocol).

#### Ufunc execution protocol

The following table defines the scope of VTL's shared callback-based helpers.
“Supported” describes only the stated helper contract; “Partial” means a
focused equivalent exists but does not implement NumPy's full ufunc-object
protocol; “Unsupported” means no shared API is currently exposed.

| NumPy protocol | VTL status | Current contract |
|---------------|------------|------------------|
| Operand broadcasting | Supported | `ufunc_binary` broadcasts operands and allocates the result; `ufunc_binary_into` requires `out` to match the broadcast result shape. Inputs use one shared element type. |
| Strided input/output views | Supported | Unary/binary helpers and axis reductions use logical tensor order; caller-owned outputs may be strided views. |
| `out=` | Supported | `ufunc_unary_into` and `ufunc_binary_into` write to and reuse caller-owned output tensors of the exact result shape. |
| `where=` mask | Supported | `where` must broadcast to the result shape without expanding it; false positions preserve the supplied output's prior values. |
| `casting=` | Unsupported | These helpers require the same `T` for operands and output. NumPy's per-call casting modes are not exposed; promoted arithmetic is covered separately by #717. |
| Output aliasing and overlap | Supported | Unary/binary into paths calculate selected values before writing, so overlapping input/output views use the original input values. |
| Empty inputs | Supported | Flat reduce returns its explicit identity; axis reduce returns that identity for empty reduced axes; accumulate preserves shape; reduceat rejects indices into empty inputs. |
| `ufunc.reduce` | Partial | `ufunc_reduce` and `ufunc_reduce_axis` require an explicit identity; the flat variant follows logical row-major order. NumPy's optional initial/keepdims protocol is not exposed. |
| `ufunc.accumulate` | Partial | Flat and axis variants require an explicit identity; axis scans preserve input shape. |
| `ufunc.reduceat` | Partial | `ufunc_reduceat` reduces flattened logical segments, including NumPy's non-increasing-start rule; axis-wise segmented reduction is not exposed. |
| `ufunc.outer` | Partial | `ufunc_outer` applies a callback to every pair of flattened logical values and returns a matrix; NumPy's general ufunc object protocol is not exposed. |
| `ufunc.at` | Partial | `ufunc_at` mutates flattened logical positions sequentially, applies repeated indices in order, and validates all indices before mutation. |

Promoted tensor-to-tensor result dtypes are:

| Tensor operands | Add/subtract/multiply | Power | Float power | True divide | Floor divide | Fmod | Remainder |
|-----------------|-----------------------|-------|-------------|--------------|--------------|------|-----------|
| Real numeric pair | `promote_types` | `promote_types` | f64 | f32 if promoted dtype is f32; else f64 | `promote_types` | `promote_types` | `promote_types` |
| Complex128 with numeric, bool, or complex128 | complex128 | unsupported | unsupported | complex128 | unsupported | unsupported | unsupported |
| Boolean pair | unsupported | int8 | f64 | f64 | unsupported | unsupported | unsupported |

`pow_promoted` rejects 64-bit integer operands when `promote_types` selects a
floating output, and rejects negative exponents when the result dtype is
integer. Boolean pairs return `int8`, matching NumPy's result dtype.

The `add_scalar_promoted`, `subtract_scalar_promoted`,
`multiply_scalar_promoted`, `divide_scalar_promoted`,
`remainder_scalar_promoted`, and `fmod_scalar_promoted` variants, as well as
the corresponding `floor_divide_scalar_promoted` form, apply NumPy-style
weak scalar promotion with the tensor as the left operand. Matching
`*_scalar_left_promoted` variants put the scalar on the left, preserving
noncommutative operation order. These APIs support real numeric and complex
tensors, except remainder, fmod, and floor division, which reject complex
values. Real float arrays retain their dtype;
integer arrays retain their dtype for in-range integer/bool scalars and promote
to float64 for floating scalars; integer true division returns float64.
Complex arrays retain complex128 with real or complex scalars, while a complex
scalar promotes real numeric and boolean arrays to complex128. Out-of-range
integer scalars return an error. Complex remainder, fmod, and floor division
are unsupported. The output type parameter is explicit because V requires
statically typed generic return values; the implementation checks it against
the inferred result dtype.
Floor division preserves the promoted dtype, rounds toward negative infinity,
returns zero for integer zero divisors, and follows IEEE behavior for floating
zero divisors.

Weak-scalar result dtypes for the supported arithmetic are:

| Array dtype | Scalar dtype | Add/subtract/multiply | True divide | Floor divide | Fmod | Remainder |
|-------------|--------------|-----------------------|-------------|--------------|------|-----------|
| f32 / f64 | real numeric or bool | array dtype | array dtype for floats | array dtype | array dtype | array dtype |
| integer | in-range integer or bool | array dtype | f64 | array dtype | array dtype | array dtype |
| integer | real float | f64 | f64 | f64 | f64 | f64 |
| real numeric or bool | complex128 | complex128 | complex128 | unsupported | unsupported | unsupported |
| complex128 | real numeric, bool, or complex128 | complex128 | complex128 | unsupported | unsupported | unsupported |

An out-of-range integer scalar is an error rather than an implicit wrap.
`round_decimals` rounds real floating-point tensors to positive or negative
decimal places using ties-to-even, matching NumPy `round` for the supported
same-dtype inputs; integer tensors are copied unchanged. VTL preserves finite
inputs rather than overflowing when an extreme positive decimal position
would overflow its intermediate scale.

Weak-scalar extrema are exposed as `maximum_scalar_promoted`,
`minimum_scalar_promoted`, `fmax_scalar_promoted`, and `fmin_scalar_promoted`,
with corresponding `*_scalar_left_promoted` forms. Floating arrays retain
their dtype; integer arrays retain their dtype for in-range integer or boolean
scalars and promote to f64 for floating scalars. Boolean arrays with boolean
scalars remain boolean; other numeric scalars use `promote_types`. Complex
inputs produce complex128. `maximum`/`minimum` propagate NaNs, while `fmax`/
`fmin` ignore one NaN and return the left operand when both are NaN. For
opposite-sign real zero ties, `fmax` selects positive zero and `fmin` selects
negative zero regardless of operand order; same-sign zero ties retain their
sign. `maximum`/`minimum` retain their ordinary operand-order tie behavior.
The output type is explicit and validated against the inferred result dtype.
`nan_to_num` replaces NaN with zero and infinities with finite type limits for
f32/f64 tensors and replaces non-finite real and imaginary components
independently for complex128 tensors, with optional custom replacements.
Integer tensors are copied. For real floating-point tensors,
`Variable.nan_to_num` also
propagates gradients through finite inputs and returns zero gradients for
replaced NaN/infinite inputs, matching PyTorch.
See [NumPy's nan_to_num reference](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
for the component-wise complex behavior.
`signbit` returns a boolean tensor for negative floating-point sign bits,
including negative zero; signed negative integers are true, while unsigned
integers and booleans produce false. `copysign` applies the sign bit of one
f32/f64 tensor to another tensor's magnitude with broadcasting. `logaddexp` and
`logaddexp2` use stable log-domain formulas, preserve f32/f64 dtype, and
broadcast same-dtype operands while handling IEEE infinities and NaNs. See the
[math ufunc tutorial](./TUTORIAL_MATH_UFUNCS.md) and
[`math_log_domain` example](../examples/math_log_domain/README.md).
`Tensor.modf` returns fractional then integral parts, matching NumPy's output
order, and `Tensor.frexp` returns floating-point mantissas with integer base-2
exponents. The `ldexp` function reconstructs values and broadcasts mantissas
against integer exponent tensors. `ldexp_scalar` and `ldexp_scalar_left` also
cover scalar exponents and mantissas. These decomposition ufuncs support f32/f64;
see the [math ufunc tutorial](./TUTORIAL_MATH_UFUNCS.md) and
[`math_decomposition` example](../examples/math_decomposition/README.md).
Elementwise `max`/`min` match NumPy's NaN-propagating `maximum`/`minimum`, and
`fmax`/`fmin` ignore a single NaN operand. These operations broadcast and
support boolean, integer, floating-point, and complex tensors; floating-point
zero ties follow each NumPy ufunc's sign behavior.
`Tensor.abs()` preserves integer values without converting through `f64`;
the minimum signed integer retains its fixed-width value, matching NumPy.
`fmod` follows NumPy's sign-of-dividend remainder rule. Integer inputs use
exact-width arithmetic, including values beyond float64's exact-integer range;
integer zero divisors and signed minimum modulo `-1` produce zero. Floating
inputs follow IEEE/libm behavior and preserve the sign of zero remainders.

Unary transcendental operations (`exp`, `sin`, `log`, and 20 more) use a
contiguous fast path that calls libm directly per element for f32/f64
(f32 is evaluated in f64 and rounded once), avoiding the previous
per-element closure and sum-type boxing. On this machine (1M f64, `-prod`)
`exp` improved from 12.1 ms to 4.7 ms per iteration (NumPy: ~3.3 ms) and
`sin` measures 15.5 ms against NumPy's ~12.8 ms.

**Remaining:** audit unary/binary function families and complete the missing
NumPy ufunc coverage in [#777](https://github.com/vlang/vtl/issues/777).
Promoted arithmetic still lacks broader operation coverage. Automatic
output-dtype selection is not provided: callers select the statically typed
result parameter, which VTL validates against the promotion contract in
[#717](https://github.com/vlang/vtl/issues/717).

### Reductions and statistics

**Current:** single-axis and multi-axis sum/product; NaN-aware
sum/product/min/max, with typed `_as` forms for NaN-aware extrema; logical
all/any with `keepdims`; weighted average globally, over
one axis, and over multiple axes with optional `keepdims`; scalar
population/sample variance and standard deviation; single-axis and multi-axis
mean/variance/std with explicit `keepdims` and NaN-aware forms; explicit
accumulator dtypes for global, single-axis, and multi-axis regular sum/product
and NaN-aware sum/product; axis-wise
`trapezoid`; all thirteen NumPy quantile estimators via `QuantileMethod`, with
global, axis, multi-axis, multi-quantile, percentile, and NaN-aware forms;
axis-wise argmax/argmin with `*_axis_with_keepdims` variants (legacy APIs
retain the axis) and squeezed min/max/argmin/argmax convenience results; median
and NaN-aware median helpers over one or multiple axes with explicit
`keepdims`; multi-axis min/max;
global flattened `argmin`/`argmax` and `nanargmin`/`nanargmax` index tensors
with optional `keepdims`, retaining every input dimension as length one;
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
floating-point slices produce NaN. `nanmin_as`/`nanmax_as` and their axis and
multi-axis forms compare source values before converting to the requested output
type, preserving exact integer extrema when the selected output type can hold
them. Empty or all-NaN results require a floating-point output type to represent
NaN; the legacy extrema APIs continue to return `f64`. Weighted averages expose typed global,
single-axis, and multi-axis forms through `average_as` and its axis variants.
`bincount_as` and `bincount_weighted_as` likewise expose output/accumulator
type selection for unweighted and weighted label counts.
`trapezoid_as`, `trapezoid_axis_as`, and `trapezoid_x_axis_as` select an
`f32` or `f64` integration accumulator/output type.
`stats.sum_accurate` provides an opt-in Neumaier-compensated global sum for
floating-point tensors. `Tensor.cumsum_flat` and `Tensor.cumprod_flat` provide
NumPy's flattened `axis=None` cumulative behavior, with explicit accumulator
types through `cumsum_flat_as` and `cumprod_flat_as`; their `Variable`
counterparts propagate gradients through the flattened order. NaN-aware
`nancumsum`/`nancumprod` support axis and flattened scans, skip NaN inputs using
the additive or multiplicative identity, and provide typed `_as` variants.

**Remaining:** add `keepdims` options consistently across the remaining
reduction families, extend accumulator dtype controls to other reductions, and
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
| `sum` / `product` | G; KD via `sum_with_keepdims` / `prod_with_keepdims`; A/M; A=() is identity | T; `_as` selects U | Propagate | Identities 0 / 1 |
| `nansum` / `nanprod` | G; KD via `nansum_with_keepdims` / `nanprod_with_keepdims`; A/M | f64; `_as` selects U | Skip | Identities 0 / 1 |
| `min` / `max` | G/A/M; A retains by default; `*_axis_with_keepdims`; M takes KD | T | Propagate first NaN | G returns 0; empty reduced A errors |
| `nanmin` / `nanmax` | G/A/M; retained/squeezed A; `*_axis_with_keepdims` | f64; `_as` selects U | Skip | No valid values -> NaN; `_as` U must be float |
| `mean` / `variance` / `std` | G; KD via `*_with_keepdims`; A/M; KD on A/M | mean T; var/std f64; `_as` selects U | Propagate | See moments details below |
| `nanmean` / `nanvar` / `nanstd` | G; KD via `*_with_keepdims`; A/M; KD on A/M | f64; `_as` selects U | Skip | NaN if empty, invalid, or under-sampled |
| `argmin` / `argmax` | G flat; KD `*_flat_with_keepdims`; A `*_axis_with_keepdims` | int | First NaN; `nanarg*` skips | Empty/no-valid reductions error |
| `median` / quantiles | G/A/M; KD forms; multi-q prepends q axis | f64 | Ordinary forms propagate; `nan*` skips | Median NaN; quantile errors |
| Weighted average / quantiles | G/A/M; KD; multi-q adds q axis | f64; weighted-average `_as` selects U | Average propagates; `nan` skips | Empty/zero-weight errors |
| `bincount` | G over vector; no KD | int; `_as` selects U; weighted defaults to f64 | Integer labels; weight NaNs propagate | Empty vector returns `minlength` zeros |
| `all` / `any` | G/A/M; KD; A=() is elementwise | bool | NaN is truthy | Identities true / false |
| `cumsum` / `cumprod` | A preserves shape; flat returns 1-D; no KD | T; `_as` selects U | Propagates forward | Empty tensor |
| `nancumsum` / `nancumprod` | A preserves shape; flat returns 1-D; no KD | T; `_as` selects U | Skip NaN as 0 / 1 | Empty tensor |
| `trapezoid` | Last/A axis; reduced axis removed; no KD | f64; `_as` selects f32/f64 | Propagates | Empty/singleton axis integrates to 0 |
| `ptp` | G/A/M; KD on tensor forms | T | Propagate | Empty reduced extent errors |

Legend: G=global, A=one axis, M=multiple axes, KD=`keepdims`, T=input type,
U=explicit output/accumulator type, q=number of requested quantiles.

For moments, legacy global `mean` returns the additive identity on an empty
real tensor; `mean_as[T, f64]` returns NaN. Global variance/std error when the
observation count does not exceed `ddof`; axis variance/std return NaN for
under-sampled slices. NaN-aware moments return NaN for empty/all-NaN inputs or
slices and when valid observations do not exceed `ddof`.

Axis `argmin`/`argmax` keep the reduced dimension by default; use
`*_axis_with_keepdims(..., false)` or `*_axis_squeeze` to remove it. Ordinary
quantile forms propagate NaNs in covered tests;
ordinary median propagates NaNs globally and for axis/multi-axis slices;
`nan*` forms skip them. Empty median returns NaN, while empty quantile/percentile
calls error. Weighted average errors on empty inputs or zero total weight;
weighted quantiles use the inverted-CDF method and error if valid observations
have zero total weight. Ordinary `min_axis`/`max_axis` retain the reduced axis
for compatibility; use their `*_axis_with_keepdims(..., false)` variants to
remove it. NaN-aware extrema retain it in the legacy `*_axis_keepdims` forms;
their `*_axis_with_keepdims` variants select either shape explicitly.

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
f64 complex return type. Tests cover strided real f64/f32 and complex f32
inputs, including offset views, against their contiguous logical-order
equivalents. Paired real and complex FFT benchmarks now cover contiguous and
stride-2 inputs at four sizes, with analytic error checks and repeated timings;
the benchmark README records a reproducible 16,384-element comparison.

**Remaining:** extend performance evidence to batched/N-D transforms and finish
the dtype-promotion audit across all public FFT wrappers.

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
The generator's `vonmises(location, concentration, shape)` returns wrapped
angles in `[-pi, pi)` and uses Best-Fisher rejection sampling for concentration
through `1e16`; concentrations at or below `1e-150` use the uniform limit to
avoid underflow.

**Remaining:** add other less common NumPy distributions and
define reproducibility across runtime versions.

### Input and output

**Current:** model serialization and dataset loaders; numeric CSV tensors with
configurable delimiters/comments, headers, missing-value fills, and explicit
element-type parsing; typed `.npy` and named `.npz` I/O with mixed supported
dtypes. `.npy` accepts v1/v2/v3 headers, both byte orders, Fortran order,
bool, f32/f64, complex128, and signed or unsigned integer widths when the
requested V type matches. Read-only `.npy` mappings expose leading-axis row
ranges as owned tensors without decoding the full payload at open.
`.npz` read/write also supports complex128 members.

**Remaining:** support more text formats and dtypes, add optional HDF5
interoperability, and continue rejecting unsupported object/structured values.
Mapped reads currently copy the requested row range and do not support writes.
See [#779](https://github.com/vlang/vtl/issues/779).

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

**Remaining:** provide an output-type dispatch API that does not require the
caller to specify the generic result type, extend weak scalar promotion to
additional scalar categories, extend promoted arithmetic coverage across
supported operations and string conversion, and extend complex support beyond
elementwise operations, `matmul`, Einstein contractions, `solve_complex`,
`inv_complex`, `det_complex`, `slogdet_complex`, and `lstsq_complex` to more
general linear algebra and random generation, add complex32, and define integer
overflow and structured-data limits.

### Masked and missing data

**Current:** `MaskedArray[T]` pairs values with a broadcastable boolean mask
(`true` means missing). It supports broadcasted arithmetic and comparisons
with mask union, scalar arithmetic, filling masked entries, compression,
valid-value counting, mixed indexing, and axis-wise gather/update that keeps
values and masks aligned. Sum, product, min, max, mean, weighted average,
variance, standard deviation, median, all thirteen NumPy quantile estimators
and their percentile forms, and masked-aware argmin/argmax work globally,
along one axis, or along multiple axes. Axis and multi-axis reductions support
`keepdims`; negative axes are
normalized and repeated or out-of-range axes are rejected. Arg reductions
return the first valid matching index and mark all-masked slices in the output
mask. Axis reductions mark empty slices in the output mask;
global reductions expose an `is_masked` flag when every value is missing.
Cumulative sum and product work over flattened logical order or along one axis;
masked positions stay masked and contribute their operation's identity to the
running accumulator. Explicit output accumulator types are available for both
operations. See the [masked data tutorial](./TUTORIAL_MASKED_DATA.md).

**Remaining:** additional masked statistics and interoperability with
autograd and neural-network operations. NaN-aware
reductions are separate and do not automatically treat NaN as masked data.

### Structured and record arrays

**Current:** not supported as a general tensor feature.

**Proposed boundary:** keep heterogeneous records outside `Tensor[T]` and
place future table conversion in a companion package. Conversion should select
named fields explicitly into homogeneous tensors, preserve row order, and
reject unsupported layouts or implicit casts. This proposal awaits maintainer
review; see [ADR-0001](./adr/0001-structured-array-boundary.md).

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

**Current:** deterministic CPU examples cover polynomial fitting, differentiable
functional transforms, mini-batch classification, and inference checkpoints.
The [NumPy and classical ML migration tutorial](./TUTORIAL_NUMPY_ML_MIGRATION.md)
links each workflow to a runnable example, expected output, and its supported
semantics.

**Remaining:** broaden classical ML evaluation beyond the small polynomial
example and define deployment formats beyond the limited CPU inference
checkpoint. These examples do not establish full NumPy, scikit-learn, or
production-deployment parity.

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

[i726]: https://github.com/vlang/vtl/issues/726
[i779]: https://github.com/vlang/vtl/issues/779
