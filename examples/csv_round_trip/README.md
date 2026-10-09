# CSV tensor round trip

Write a 2 by 2 tensor with a CSV header, read it back while skipping that header, and parse a record without a final newline.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/csv_round_trip/main.v
```

## Notes

Temporary files are created under the system temporary directory and removed when the program exits.
