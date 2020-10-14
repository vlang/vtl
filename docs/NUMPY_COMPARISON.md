# Basic Usage

```sh
>>> import vnum.num
>>> num.seq(30)
[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
```

`vnum` provides vectorized operations on ndarrays.

```sh
>>> a := num.seq(12).reshape([3, 2, 2])
>>> num.sum_axis(a, 1)
[[ 2,  4],
 [10, 12],
 [18, 20]]
```

Use the `vnum.linalg` module for powerful `BLAS` backed routines.

```sh
>>> import vnum.num
>>> import vnum.la
>>> a := num.seq(60).reshape([3, 4, 5])
>>> b := num.seq(24).reshape([4, 3, 2])
>>> res := la.tensordot(a, b, [1, 0], [0, 1])
>>> res
[[4400, 4730],
 [4532, 4874],
 [4664, 5018],
 [4796, 5162],
 [4928, 5306]]
```

## For Numpy Users

<table>
   <tr>
      <th>NumPy </th>
      <th>Vnum</th>
   </tr>
   <tr>
      <td>
         <code>
         np.array([[1.,2.,3.], [4.,5.,6.]])
         </code>
      </td>
      <td>
         <code>
         num.from_int([1, 2, 3, 4, 5, 6], [2, 3])
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
         num.seq(10)
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
         num.linspace(0, 10, 11)
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
         num.ones([3, 4, 5])
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
         num.zeros((3, 4, 5))
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
         num.zeros([3, 4, 5]).copy('F')
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
         num.full([3, 4], 7)
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
         a.get([-1])
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
         a.get([1, 4])
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
         num.sum_axis(1)
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
         num.diag(a)
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
         num.concatenate([a, b], 1)
         </code>
      </td>
   </tr>
</table>

# Numpy compared with Vlang VNUM

This is a collection of common exercises using numpy, and their equivalent solutions using vnum

#### 1. Import the numpy package under the name `np`

```python
import numpy as np
```

```v
import vnum.num
```

#### 2. Print the numpy version and the configuration

```python
print(np.__version__)
np.show_config()
```

```v
import vnum
println(vnum.VERSION)
```

#### 3. Create a null vector of size 10

```python
Z = np.zeros(10)
print(Z)
```

```v
a := num.zeros([10])
```

#### 4. Create a null vector of size 10 but the fifth value which is 1

```python
Z = np.zeros(10)
Z[4] = 1
print(Z)
```

```v
z := num.zeros([10])
z.set([4], 1)
println(z)
```

#### 5. Create a vector with values ranging from 10 to 49

```python
Z = np.arange(10,50)
print(Z)
```

```v
z := num.seq_between(10, 50)
println(z)
```

#### 6. Reverse a vector (first element becomes last)

```python
Z = np.arange(50)
Z = Z[::-1]
print(Z)
```

```v
z := num.seq(50)
r := z.slice([[0, 50, -1]])
println(z)
```

#### 7. Create a 3x3 matrix with values ranging from 0 to 8

```python
nz = np.arange(9).reshape(3, 3)
print(nz)
```

```v
nz := num.seq(9).reshape([3, 3])
```

#### 8. Create a 3x3 identity matrix

```python
Z = np.eye(3)
print(Z)
```

```v
z := num.eye(3)
println(z)
```

#### 9. Create a 3x3x3 array with random values

```python
Z = np.random.random((3,3,3))
print(Z)
```

```
z := num.random(0, 1, [3, 3, 3])
println(z)
```

#### 10. Create a 10x10 array with random values and find the minimum and maximum values

```python
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)
```

```v
z := num.random(0, 1, [10, 10])
zmin := z.min()
zmax := z.max()
println(zmin, zmax)
```

#### 11. Create a random vector of size 30 and find the mean value

```python
Z = np.random.random(30)
m = Z.mean()
print(m)
```

```v
z := num.random(0, 1, [30])
m := z.mean()
println(m)
```

#### 12. Create a 2d array with 1 on the border and 0 inside

```python
Z = np.ones((10,10))
Z[1:-1,1:-1] = 0
print(Z)
```

```v
z := num.ones([10, 10])
z.slice([[1, -1], [1, -1]]).fill(0)
println(z)
```
