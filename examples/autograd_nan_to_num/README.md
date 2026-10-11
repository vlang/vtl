# Autograd and `nan_to_num`

Replace non-finite tensor values before computing a loss. Backpropagation sends
gradients through finite inputs and gives replaced NaN or infinite inputs a
zero gradient, following PyTorch's `nan_to_num` derivative.

From `~/.vmodules`, compile and run separately:

```bash
VJOBS=2 v -o /tmp/vtl-autograd-nan-to-num ./vtl/examples/autograd_nan_to_num/main.v
/tmp/vtl-autograd-nan-to-num
```

Expected output:

```text
cleaned values: [2.0, 0.0, 10.0, -10.0]
input gradient: [1.0, 0.0, 0.0, 0.0]
```
