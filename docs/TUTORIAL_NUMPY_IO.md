# NumPy `.npy` and `.npz` input and output

`vtl.npy` reads and writes uncompressed NumPy `.npy` arrays. Files can use
versions 1.0, 2.0, or 3.0. The reader handles C and Fortran order and converts
big-endian data to VTL's logical row-major order.

The file dtype must match the requested V type. Supported types are `bool`,
`f32`, `f64`, signed `i8`/`i16`/`i32`/`i64`/`int`, and unsigned
`u8`/`u16`/`u32`/`u64`. V's native `int` uses the current platform width.
Object, string, structured, and complex dtypes are rejected.

```v
import vtl
import vtl.npy
import os

path := os.join_path(os.temp_dir(), 'matrix.npy')
defer {
	os.rm(path) or {}
}
matrix := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [2, 2])!
npy.write(path, matrix)!
loaded := npy.read[f64](path)!
assert loaded.shape == [2, 2]
assert loaded.to_array() == [1.0, 2.0, 3.0, 4.0]
```

The `write` function emits `.npy` v1.0 with little-endian numeric data.
See [the runnable example](../examples/npy_round_trip) for a complete program.

## Numeric CSV tensors

`vtl.csv` reads numeric fields into a rank-two tensor and writes rank-two
tensors as CSV. It supports quoted fields through V's standard CSV parser,
configurable delimiters and comments, headers, skipped rows, selected columns,
and a row limit. Read each tensor with its expected V element type; CSV itself
does not store dtype or shape metadata.

```v
import vtl.csv

features := csv.read[f64]('training.csv', csv.CsvReadConfig{
	skip_header: true
	use_cols:    [0, 2]
	max_rows:    1000
})!
csv.write('features.csv', features, csv.CsvWriteConfig{
	header: ['height', 'weight']
})!
```

See [the CSV round-trip example](../examples/csv_round_trip) for a runnable
program.

## Compressed `.npz` archives

`vtl.npz` reads compressed and uncompressed ZIP archives and loads a named
`.npy` member without extracting archive paths to disk. Members can have
different supported dtypes; select the matching V type when reading each
member. `write` stores same-type tensors efficiently, while `write_arrays`
accepts tensors wrapped with `npz.array` for mixed-dtype archives.

```v
import vtl
import vtl.npz

arrays := {
	'features': vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [2, 2])!
	'targets':  vtl.from_1d[f64]([0.0, 1.0])!
}
npz.write('training.npz', arrays)!
features := npz.read[f64]('training.npz', 'features')!
println(npz.members('training.npz')!)

mixed := {
	'features': npz.array[f64](vtl.from_1d[f64]([1.0, 2.0])!)
	'labels':   npz.array[i32](vtl.from_1d[i32]([0, 1])!)
}
npz.write_arrays('mixed_training.npz', mixed)!
```

See [the runnable `.npz` example](../examples/npz_round_trip) for a complete
round trip. [A second example](../examples/npz_read_compressed) reads a
NumPy-generated compressed archive. The reader and writer support mixed dtypes
by selecting a matching V type for each member.
