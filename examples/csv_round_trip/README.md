# CSV tensor round trip

Write a 2 by 2 tensor with a CSV header, read it back while skipping that header, parse a
record without a final newline, and import a NumPy-generated text fixture with a missing-value
marker and explicit fill value.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/csv_round_trip/main.v
```

## Notes

Temporary files are created under the system temporary directory and removed when the program
exits. The committed fixture uses NumPy `savetxt`-formatted scientific notation, a semicolon
delimiter, a header, and a `nan` field; its leading comment line exercises comment handling.
