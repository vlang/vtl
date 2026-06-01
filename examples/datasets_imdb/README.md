# datasets_imdb

Shape smoke for the IMDB sentiment-analysis dataset loader.

## Run

```bash
v run vtl/examples/datasets_imdb/main.v
```

The loader returns:

- `train_features`: `[25000]`
- `test_features`: `[25000]`
- `train_labels`: `[25000]`
- `test_labels`: `[25000]`

This example may download/cache dataset files depending on the local dataset
state. For CI-safe training, prefer synthetic examples.

See [datasets/README.md](../../datasets/README.md).
