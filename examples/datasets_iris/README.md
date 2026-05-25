# Iris Dataset Example

This example demonstrates loading and inspecting the classic Iris dataset via `vtl.datasets`.

## What it covers

- Dataset download/loading (`datasets.load_iris`)
- Feature/label tensor shapes
- First-row inspection
- Class distribution summary (0/1/2)

## Run

```sh
v run examples/datasets_iris/main.v
```

## Notes

- First run may download the dataset file.
- Labels are encoded as:
  - `0`: Iris-setosa
  - `1`: Iris-versicolor
  - `2`: Iris-virginica
