# One-Hot Labels

Encode class-index tensors as one-hot tensors for multiclass ML workflows.
The output appends a class dimension, accepts arbitrary input ranks, and
reports invalid class IDs instead of indexing outside the output.

Run from `~/.vmodules`:

```sh
v -o /tmp/vtl-one-hot -prod ./vtl/examples/one_hot/main.v
/tmp/vtl-one-hot
```
