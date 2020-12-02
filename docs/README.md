# V Tensor Library

## Numpy Comparison

| Numpy                                | VTL                                                 |
| ------------------------------------ | --------------------------------------------------- |
| `np.array([[1.,2.,3.], [4.,5.,6.]])` | `vtl.from_varray([1., 2., 3., 4., 5., 6.], [2, 3])` |
| `np.arange(10)`                      | `vtl.seq(10) // or vtl.range(to: 10)`               |
| `np.ones((3, 4, 5))`                 | `vtl.ones([3, 4, 5])`                               |
| `np.zeros((3, 4, 5))`                | `vtl.zeros([3, 4, 5])`                              |
| `np.full((3, 4), 7)`                 | `vtl.full([3, 4], 7.0)`                         |
| `a[-1]`                              | `a.get([-1])`                                       |
| `a[1, 4]`                            | `a.get([1, 4])`                                     |
| `a[0:5]`                             | `a.slice([0, 5])`                                   |
| `a[1:4:2]`                           | `a.slice([1, 4, 2])`                                |
| `a.T`                                | `a.t()`                                             |
| `np.diag(a)`                         | `vtl.diag(a)`                                       |
| `a[:] = 3.`                          | `a.fill(3.0)`                                   |
| `np.concatenate((a, b), axis=1)`     | `vtl.concatenate([a, b], axis: 1)`                  |
