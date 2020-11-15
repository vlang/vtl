[![Build Status](https://github.com/vlang/vtl/workflows/CI/badge.svg)](https://github.com/vlang/vtl/commits/master)

# VTL - V Tensor Library

## VTL Provides

- An n-dimensional `NdArray` data structure
- Sophisticated reduction, elementwise, and accumulation operations
- Data Structures that can easily be passed to C libraries
- Powerful linear algebra routines backed by VSL that uses LAPACKE and OpenBLAS.

In the docs you can find a complete [Numpy vs Vtl comparison](./docs/NUMPY_COMPARISON.md) 

## Installation

Because we use CV for linking [VSL](https://github.com/vlang/vsl) with many libraries, it is not enough to use the so convenient `v install` _or_ `vpkg get` functionality for installing VSL. First we need to install some dependencies in order to have VSL working as expected.

### Install dependencies (optional)

We use VSL as backend for some functionalities and [VSL](https://github.com/vlang/vsl) links with existent libraries written in C and Fortran, such as OpenBLAS and LAPACK. These existing libraries have been fundamental for the development of high-performant simulations over many years. We believe that it is nearly impossible to rewrite these libraries in native V and at the same time achieve the same speed delivered by them.

VTL requires VSL's OpenBLAS and LAPACKE wrappers. If you wish you to use vtl without these, the `vtl.num` module will still function as normal.

Follow this [install instructions](https://github.com/vlang/vsl#install-vsl-locally) at VSL docs in order to install VSL with all needed dependencies.

### Install VTL

**Via vpm**

```sh
$ v install vtl
```

**Via [vpkg](https://github.com/v-pkg/vpkg)**

```sh
$ vpkg get https://github.com/vlang/vtl
```

Done. Installation completed.

## Testing

To test the module, just type the following command:

```sh
$ make test # or ./bin/test
```

## License

[MIT](LICENSE)

## Contributors

> This project is based on the work done by Christopher ([christopherzimmerman](https://github.com/christopherzimmerman)) and the rest of the Vlang Num group.

- [Ulises Jeremias Cornejo Fandos](https://github.com/ulises-jeremias) - Core Maintainer
