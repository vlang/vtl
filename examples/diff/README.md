# Discrete Differences

Use `vtl.diff` to calculate adjacent differences along any tensor axis. The
`n` argument applies the operation repeatedly; negative axes count from the
end.

Run from `~/.vmodules`:

```sh
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 -- \
  env VJOBS=2 v -prod run ./vtl/examples/diff/main.v
```
