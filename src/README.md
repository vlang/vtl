<div align="center">
<h1>The V Tensor Library</h1>

[vlang.io](https://vlang.io) |
[Docs](https://vlang.github.io/vtl) |
[Tutorials](https://github.com/vlang/vtl/blob/main/docs/TUTORIAL.md) |
[Changelog](#) |
[Contributing](https://github.com/vlang/vtl/blob/main/CONTRIBUTING.md)

</div>

```v ignore
>>> import vtl
>>> t := vtl.from_array([1.0, 2, 3, 4], [2, 2])!
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

### Install dependencies (optional)

We use [VSL](https://github.com/vlang/vsl) as backend for some functionalities.
VTL requires VSL's linear algebra module.
If you wish you to use vtl without these, the `vtl` module will still function as normal.

Follow this [install instructions](https://github.com/vlang/vsl#install-vsl-locally)
at VSL docs in order to install VSL with all needed dependencies.

### Install VTL

```sh
v install vtl
```

Done. Installation completed.

## Testing

To test the module, just type the following command:

```sh
v test .
```

## License

[MIT](LICENSE)

## Contributors

> This work was originally based on the work done by
> Christopher ([christopherzimmerman](https://github.com/christopherzimmerman)).

> The development of this library continues its course after having reimplemented its core
> and a large part of its interface. In the same way, we do not want to stop recognizing
> the work and inspiration that the library done by Christopher has given.

<a href="https://github.com/vlang/vtl/contributors">
  <img src="https://contrib.rocks/image?repo=vlang/vtl"/>
</a>

Made with [contributors-img](https://contrib.rocks).

