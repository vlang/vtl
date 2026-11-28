# NumPy dtype policy

`vtl.dtypes` records the boundary for non-numeric NumPy dtypes tracked in
[#747](https://github.com/vlang/vtl/issues/747). VTL tensors stay homogeneous
and numeric; heterogeneous table/record data belongs in a companion package.

## Supported families

`boolean`, `integer`, `unsigned`, `float`, and `complex`. Supported widths are
`b1`, `i1/i2/i4/i8`, `u1/u2/u4/u8`, `f4/f8`, and `c16`.

## Excluded families

- `S` fixed-width bytes and `U` fixed-width unicode strings
- `M` datetime and `m` timedelta (including units such as `ns`, `us`, `ms`)
- `O` object
- `V` structured/void records

These are rejected explicitly, with the family name and a pointer to the
companion-package boundary. No value is silently coerced into a number.
`parse` recognizes excluded descriptors so diagnostics can name them rather
than reporting a generic parse failure. Datetime/timedelta descriptors must
include a unit and use width 8; the unit is preserved for diagnostics but no
datetime arithmetic is provided.

## Example

```v
descriptor := dtypes.parse('<U3')!
descriptor.ensure_supported() or {
	println(err.msg()) // names the unicode string family and the boundary
}
```

`npy.read_bytes` uses this policy, so reading a `.npy` file with a string,
datetime, object, or structured dtype fails with a specific reason.

Run the focused tests with:

```sh
v test tests/dtypes/dtype_test.v
```

Construction, indexing, comparison, selection, and serialization of supported
non-numeric types remain out of scope and tracked in #747, together with the
data-analysis example and NumPy dtype fixtures.
