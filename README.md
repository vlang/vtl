<div align="center">
<h1>The V Tensor Library</h1>

[vlang.io](https://vlang.io) |
[Docs](https://vlang.github.io/vtl) |
[Changelog](#) |
[Contributing](https://github.com/vlang/vtl/blob/main/CONTRIBUTING.md)

</div>
<div align="center">

[![Mentioned in Awesome V][awesomevbadge]][awesomevurl]
[![Build Status][workflowbadge]][workflowurl]
[![Docs Validation][validatedocsbadge]][validatedocsurl]
[![License: MIT][licensebadge]][licenseurl]

</div>

```v nofmt
>>> import vtl
>>> t := vtl.from_varray([1., 2., 3., 4.], [2, 2])
>>> t.get([1, 1])
4.0
```

## VTL Provides

- An n-dimensional `Tensor` data structure
- Sophisticated reduction, elementwise, and accumulation operations
- Data Structures that can easily be passed to C libraries
- Powerful linear algebra routines backed by VSL that uses LAPACKE and OpenBLAS.

In the [docs](https://vlang.github.io/vtl) you can find more information about this module 

## Installation

Because we use CV for linking [VSL](https://github.com/vlang/vsl) with many libraries, it is not enough to use the so convenient `v install` _or_ `vpkg get` functionality for installing VSL. First we need to install some dependencies in order to have VSL working as expected.

### Install dependencies (optional)

We use VSL as backend for some functionalities and [VSL](https://github.com/vlang/vsl) links with existent libraries written in C and Fortran, such as OpenBLAS and LAPACK. These existing libraries have been fundamental for the development of high-performant simulations over many years. We believe that it is nearly impossible to rewrite these libraries in native V and at the same time achieve the same speed delivered by them.

VTL requires VSL's OpenBLAS and LAPACKE wrappers. If you wish you to use vtl without these,
the `vtl` module will still function as normal.

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
$ ./bin/test
```

## License

[MIT](LICENSE)

## Contributors

> This work was originally based on the work done by Christopher ([christopherzimmerman](https://github.com/christopherzimmerman)) and the rest of the VLang-Num group. 

> The development of this library continues its course after having reimplemented its core
> and a large part of its interface. In the same way, we do not want to stop recognizing
> the work and inspiration that the library done by Christopher has given.

- [Ulises Jeremias Cornejo Fandos](https://github.com/ulises-jeremias) - Core Maintainer

[awesomevbadge]: https://awesome.re/mentioned-badge.svg
[workflowbadge]: https://github.com/vlang/vtl/workflows/CI/badge.svg
[validatedocsbadge]: https://github.com/vlang/vtl/workflows/Validate%20Docs/badge.svg
[licensebadge]: https://img.shields.io/badge/License-MIT-blue.svg
[awesomevurl]: https://github.com/vlang/awesome-v/blob/master/README.md#scientific-computing
[workflowurl]: https://github.com/vlang/vtl/commits/main
[validatedocsurl]: https://github.com/vlang/vtl/commits/main
[licenseurl]: https://github.com/vlang/vtl/blob/main/LICENSE
