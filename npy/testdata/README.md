# NumPy `.npy` fixtures

`numpy_complex128.npy` was generated with NumPy 2.5.3 using:

```python
import numpy as np

np.save(
    "numpy_complex128.npy",
    np.array([1.5 - 2.25j, -3.0 + 4.75j], dtype=np.complex128),
)
```

The VTL test suite reads this file directly to check interoperability with
NumPy's `complex128` descriptor and payload layout.
