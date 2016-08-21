# Description of Individual Source Code Files

The code for this package exists in six files:

## `modes.jl`

This file contains the type definitions of the finite difference modes we
provide, which are `ForwardMode`, `BackwardMode`, `CentralMode`, `ComplexMode`
and `HessianMode`. This code will be easier to understand if you delete the
documentation before reading it because the raw code is extremely simple and
the majority of the file's content is documentation.

## `step_size.jl`

This file defines the methods of the step-size function, which determines the
step-size used to compute a finite difference approximation of derivatives. The
step-size is determined by both the finite difference and the value of `x` at
which the derivative will be approximated. This code will be easier to
understand if you ignore the documentation at first, because the code is very
simple. But understanding why the code is the way it is will require that you
read some of the literature referenced in the bibliography.

## `derivative.jl`

This file defines the methods of the `derivative` function, which comes in
pure, mutating and higher-order variants. This file also contains definitions
for each mode of finite differences that's supported. The core logic for is
defined in the pure variants; the other variants are wrappers around this core
logic.

## `second_derivative.jl`

This file defines the methods of the `second_derivative` function, which comes
in pure, mutating and higher-order variants. Only one mode of finite
differences is supported. The core logic is defined in the pure variants; the
other variants are wrappers around this core logic.

## `gradient.jl`

This file defines the methods of the `gradient` function, which comes in pure,
mutating and higher-order variants. This file also contains definitions for
each mode of finite differences modes that's supported. Unlike the univariate
functions, the core logic for `gradient` is defined in the mutating variants;
the other variants are wrappers around this core logic.

## `hessian.jl`

This file defines the methods of the `hessian` function, which comes in pure,
mutating and higher-order variants. Only one mode of finite differences is
supported. Unlike the univariate functions, the core logic for `hessian` is
defined in the mutating variants; the other variants are wrappers around this
core logic.
