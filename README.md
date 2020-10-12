[![Build Status](https://github.com/vlang/vnum/workflows/CI/badge.svg)](https://github.com/vlang/vnum/commits/master)

# VNUM - V Numerical Library

## VNUM Provides

- An n-dimensional `NdArray` data structure
- Sophisticated reduction, elementwise, and accumulation operations
- Data Structures that can easily be passed to C libraries
- Powerful linear algebra routines backed by VSL that uses LAPACKE and OpenBLAS.

* * *

We use VSL as backend for some functionalities and VLS links with existent libraries written in C and Fortran, such as OpenBLAS and LAPACK. These existing libraries have been fundamental for the development of high-performant simulations over many years. We believe that it is nearly impossible to rewrite these libraries in native V and at the same time achieve the same speed delivered by them.

## Installation

Because of C dependencies and other libraries, the easiest way to work with VNUM is via Docker. Having Docker and VS Code installed, you can start developing powerful numerical simulations using VNUM in a matter of seconds. Furthermore, the best part of it is that it works on Windows, Linux, and macOS out of the box.

### Quick, containerized (recommended)

1. Install Docker
2. Install Visual Studio Code
3. Install the Remote Development extension for VS Code
4. Clone this repository
5. Create your application within a container (see gif below)

Done. And your system will remain "clean".

![](static/vscode-open-in-container.gif)

Our [Docker Image](https://hub.docker.com/repository/docker/vsl/vsl) also contains V and the V Tools for working with VS Code (or not). Below is a video showing the convenience of VS Code + the V tools + VSL.

## Install VNUM locally

Because we use CV for linking VSL with many libraries, it is not enough to use the so convenient `v install` _or_ `vpkg get` functionality for installing VSL. First we need to install some dependencies in order to have VSL working as expected.

### Install dependencies

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
