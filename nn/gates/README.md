# `vtl.nn.gates`

Autograd gates implement backward rules for neural-network layers, activations,
and losses. The submodules mirror those concerns (`activation`, `layers`, and
`loss`) and are primarily used internally by `vtl.nn.layers` and
`vtl.nn.loss`.

Applications should normally compose the public layer and loss APIs instead of
constructing gates directly. Custom gate work must preserve parent ordering,
gradient shapes, context ownership, and cached forward intermediates. Existing
finite-difference and gate regression tests are useful references.
