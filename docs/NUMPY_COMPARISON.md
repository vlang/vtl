# Basic Usage

```v
>>> import vtl
>>> vtl.range<f64>(to: 30)
[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17.,
 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.]
```

`vtl` provides vectorized operations on tensors.

```v
>>> a := vtl.range<f64>(to: 12).reshape([3, 2, 2])
>>> vtl.sum_axis(a, 1)
[[ 2,  4],
 [10, 12],
 [18, 20]]
```

Use the `vtl.la` module for powerful `VSL` backed routines.

```v
>>> import vtl
>>> import vtl.la
>>> a := vtl.range<f64>(to: 60).reshape([3, 4, 5])
>>> b := vtl.range<f64>(to: 24).reshape([4, 3, 2])
>>> res := la.tensordot(a, b, [1, 0], [0, 1])
>>> res
[[4400, 4730],
 [4532, 4874],
 [4664, 5018],
 [4796, 5162],
 [4928, 5306]]
```

Basic support for Machine Learning is being added in the `vtl.nn` module.  Here is a basic example learning the XOR function

```v
>>> import vtl.nn
>>> import vtl
>>> features := vtl.from_varray<f64>([0., 0., 0., 1., 1., 0., 1., 1.], [4, 2])
>>> labels := vtl.from_varray<f64>([0., 1., 1., 0.], [4, 1])
>>> mut m := nn.new(0.7, 10000, 3, "sigmoid")
>>> m.learn(features, labels)
>>> m.predict(features)
[[ 0.0387387],
 [  0.976217],
 [  0.976216],
 [0.00880164]]
```

## For Numpy Users

<table>
   <tr>
      <th>NumPy </th>
      <th>Vtl</th>
   </tr>
   <tr>
      <td>
         <code>
         np.array([[1., 2., 3.], [4., 5., 6.]])
         </code>
      </td>
      <td>
         <code>
         vtl.from_varray<f64>([1., 2., 3., 4., 5., 6.], [2, 3])
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         np.arange(10)
         </code>
      </td>
      <td>
         <code>
         vtl.range<f64>(to: 10)
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         np.linspace(0, 10, 11)
         </code>
      </td>
      <td>
         <code>
         vtl.linspace<f64>(0, 10, 11)
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         np.ones((3, 4, 5))
         </code>
      </td>
      <td>
         <code>
         vtl.ones<f64>([3, 4, 5])
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         np.zeros((3, 4, 5))
         </code>
      </td>
      <td>
         <code>
         vtl.zeros<f64>([3, 4, 5])
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         np.zeros((3, 4, 5), order='F')
         </code>
      </td>
      <td>
         <code>
         vtl.zeros([3, 4, 5]).copy('F')
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         np.full((3, 4), 7)
         </code>
      </td>
      <td>
         <code>
         vtl.full<f64>([3, 4], 7.)
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         a[-1]
         </code>
      </td>
      <td>
         <code>
         a.get<f64>([-1])
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         a[1, 4]
         </code>
      </td>
      <td>
         <code>
         a.get<f64>([1, 4])
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         a[1]
         </code>
      </td>
      <td>
         <code>
         a.slice([[1]])
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         a[0:5]
         </code>
      </td>
      <td>
         <code>
         a.slice([[0, 5]])
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         a[1:4:2]
         </code>
      </td>
      <td>
         <code>
         a.slice([[1, 4, 2]])
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         a.T
         </code>
      </td>
      <td>
         <code>
         a.t()
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         mat1.dot(mat2)
         </code>
      </td>
      <td>
         <code>
         la.matmul(mat2, mat2)
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         np.sum(a, axis=1)
         </code>
      </td>
      <td>
         <code>
         vtl.sum_axis(1)
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         np.diag(a)
         </code>
      </td>
      <td>
         <code>
         vtl.diag(a)
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         a[:] = 3
         </code>
      </td>
      <td>
         <code>
         a.fill(3)
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         a[:] = b
         </code>
      </td>
      <td>
         <code>
         a.assign(b)
         </code>
      </td>
   </tr>
   <tr>
      <td>
         <code>
         np.concatenate((a, b), axis=1)
         </code>
      </td>
      <td>
         <code>
         vtl.concatenate([a, b], 1)
         </code>
      </td>
   </tr>
</table>

# Numpy compared with Vlang VTL

This is a collection of common exercises using numpy, and their equivalent solutions using vtl

#### 1. Import the numpy package under the name `np`

```python
>>> import numpy as np
```

```v
>>> import vtl
```

#### 2. Print the numpy version and the configuration

```python
>>> print(np.__version__)
>>> np.show_config()
```

```v
>>> import vtl
>>> print(vtl.version)
```

#### 3. Create a null vector of size 10

```python
>>> Z = np.zeros(10)
>>> print(Z)
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```

```v
>>> a := vtl.zeros<f64>([10])
>>> print(a)
[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
```

#### 4. Create a null vector of size 10 but the fifth value which is 1

```python
>>> Z = np.zeros(10)
>>> Z[4] = 1
>>> print(Z)
[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
```

```v
>>> mut z := vtl.zeros<f64>([10])
>>> z.set<f64>([4], 1)
>>> print(z)
[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
```

#### 5. Create a vector with values ranging from 10 to 49

```python
>>> Z = np.arange(10, 50)
>>> print(Z)
[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33
 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]
```

```v
>>> z := vtl.range<f64>(from: 10, to: 50)
>>> print(z)
[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 
28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 
46, 47, 48, 49]
```

#### 6. Reverse a vector (first element becomes last)

```python
>>> Z = np.arange(50)
>>> Z = Z[::-1]
>>> print(Z)
[49 48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26
 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2
  1  0]
```

```v
>>> z := vtl.range<f64>(to: 50)
>>> r := z.slice([[0, 50, -1]])
>>> print(z)
[49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26,
 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
  1, 0]
```

#### 7. Create a 3x3 matrix with values ranging from 0 to 8

```python
>>> nz = np.arange(9).reshape(3, 3)
>>> print(nz)
[[0 1 2]
 [3 4 5]
 [6 7 8]]
```

```v
>>> nz := vtl.range<f64>(to: 9 .reshape([3, 3])
>>> print(nz)
[[0, 1, 2], 
[3, 4, 5], 
[6, 7, 8]]
```

#### 8. Create a 3x3 identity matrix

```python
>>> Z = np.eye(3)
>>> print(Z)
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

```v
>>> z := vtl.eye(3, 3, 0)
>>> print(z)
[[1, 0, 0], 
[0, 1, 0], 
[0, 0, 1]]
```

#### 9. Create a 3x3x3 array with random values

```python
>>> Z = np.random.random((3,3,3))
>>> print(Z)
[[[0.03764605 0.78824465 0.81525183]
  [0.47834326 0.58426438 0.66548865]
  [0.00866438 0.32231888 0.94869011]]

 [[0.21189378 0.97698459 0.27519104]
  [0.9509932  0.3371184  0.40193393]
  [0.97053465 0.78509551 0.77321058]]

 [[0.49872957 0.91425013 0.2524011 ]
  [0.22086304 0.53331973 0.69934665]
  [0.7333267  0.95177294 0.85193637]]]
```

```
>>> z := vtl.random(0, 1, [3, 3, 3])
>>> print(z)
[[[   0.17003,  0.0639199,    0.96962], 
 [  0.794206, 0.00541103,   0.384699], 
 [  0.459102,    0.39152,   0.316201]], 

[[  0.640773,   0.182407,   0.916281], 
 [   0.31492,   0.441845,   0.609481], 
 [  0.501642,   0.901246,   0.325363]], 

[[  0.385345,   0.768257,   0.495268], 
 [  0.786213,   0.106995,   0.127018], 
 [  0.347216,    0.07541,   0.605519]]]
```

#### 10. Create a 10x10 array with random values and find the minimum and maximum values

```python
>>> Z = np.random.random((10,10))
>>> Zmin, Zmax = Z.min(), Z.max()
>>> print(Zmin, Zmax)
```

```v
>>> z := vtl.random(0, 1, [10, 10])
>>> zmin, zmax := z.min(), z.max()
>>> print(zmin, zmax)
```

#### 11. Create a random vector of size 30 and find the mean value

```python
>>> Z = np.random.random(30)
>>> m = Z.mean()
>>> print(m)
```

```v
>>> z := vtl.random(0, 1, [30])
>>> m := z.mean()
>>> print(m)
```

#### 12. Create a 2d array with 1 on the border and 0 inside

```python
>>> Z = np.ones((10,10))
>>> Z[1:-1,1:-1] = 0
>>> print(Z)
[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
```

```v
>>> z := vtl.ones([10, 10])
>>> z.slice([[1, -1], [1, -1]]).fill(0)
>>> print(z)
[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```
