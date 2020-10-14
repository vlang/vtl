[![Build Status](https://github.com/vlang/vnum/workflows/CI/badge.svg)](https://github.com/vlang/vnum/commits/master)

# VNUM - V Numerical Library

## VNUM Provides

- An n-dimensional `NdArray` data structure
- Sophisticated reduction, elementwise, and accumulation operations
- Data Structures that can easily be passed to C libraries
- Powerful linear algebra routines backed by VSL that uses LAPACKE and OpenBLAS.

* * *

We use VSL as backend for some functionalities and VSL links with existent libraries written in C and Fortran, such as OpenBLAS and LAPACK. These existing libraries have been fundamental for the development of high-performant simulations over many years. We believe that it is nearly impossible to rewrite these libraries in native V and at the same time achieve the same speed delivered by them.

## Installation

## Install VNUM locally

Because we use CV for linking VSL with many libraries, it is not enough to use the so convenient `v install` _or_ `vpkg get` functionality for installing VSL. First we need to install some dependencies in order to have VSL working as expected.

### Install dependencies

VNUM requires VSL's OpenBLAS and LAPACKE wrappers. If you wish you to use vnum without these, the `vnum.num` module will still function as normal.

Follow this [install instructions](https://github.com/vlang/vsl#install-vsl-locally) at VSL docs in order to install VSL with all needed dependencies.

### Install VNUM

**Via vpm**

```sh
$ v install vnum
```

**Via [vpkg](https://github.com/v-pkg/vpkg)**

```sh
$ vpkg get https://github.com/vlang/vnum
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

> This project is based on the work done by Christopher and the rest of the Vlang Num gruop.
> 
- [Ulises Jeremias Cornejo Fandos](https://github.com/ulises-jeremias) - Core Maintainer
