FiniteDiff.jl
=============

# Introduction

The FiniteDiff package provides functions for computing approximate first and
second derivatives of univariate and multivariate functions. It uses
finite-differencing to generate these approximations, which means that it is
possible to compute an approximate derivative for any Julia function.

Despite being universally applicable, the quality of the approximated
derivatives computed by this package varies across functions and their domains.
If your function is amenable to other techniques (such as forward-mode
automatic differentiation), you will likely get better results from those
techniques. Use finite-differencing only when you need approximate derivatives
and other techniques fail.

This package is intended to be a replacement for the finite-differencing
functionality found in the Calculus package. It attempts to unify the API
for finite-differencing with the API for automatic differentiation found in the
[ForwardDiff.jl package](http://www.juliadiff.org/ForwardDiff.jl/index.html).

# API

Most users will want to work with a limited set of basic functions:

* `derivative()`: Use this for functions from R to R
* `second_derivative()`: Use this for functions from R to R
* `gradient()`: Use this for functions from R^n to R
* `hessian()`: Use this for functions from R^n to R

All functions also come in mutating variants, which offer substantially better
performance in many settings.

# Usage Examples

First, let's look at the non-mutating API:

```jl
import FiniteDiff

# Compute approximate derivative and compare it with the true value.
d = FiniteDiff.derivative(sin, 0.0)
d, cos(0.0)

# Compute approximate second derivative and compare it with the true value.
d2 = FiniteDiff.second_derivative(sin, 1.0)
d2, -sin(1.0)

# Compute approximate gradient and compare it with the true value.
gr = FiniteDiff.gradient(x -> sin(x[1]), [0.0])
gr, [cos(0.0)]

# Compare approximate Hessian and compare it with the true value.
H = FiniteDiff.hessian(x -> sin(x[1]) + cos(x[2]), [1.0, 1.0])
H, [-sin(1.0) 0.0; 0.0 -cos(1.0)]
```

Now let's look at the parallel mutating API:

```jl
import FiniteDiff

# Compute approximate derivative and compare it with the true value.
tmp1 = Array(Float64, 1)
FiniteDiff.derivative!(tmp1, sin, 0.0)
tmp1, [cos(0.0)]

# Compute approximate second derivative and compare it with the true value
tmp2 = Array(Float64, 1)
FiniteDiff.second_derivative!(tmp2, sin, 1.0)
tmp2, [-sin(1.0)]

# Compute approximate gradient and compare it with true value
tmp3 = Array(Float64, 1)
buffer = Array(Float64, 1)
FiniteDiff.gradient!(tmp3, x -> sin(x[1]), [0.0], buffer)
tmp3, [cos(0.0)]

# Compare approximate Hessian and compare it with the true value.
tmp4 = Array(Float64, 2, 2)
buffer = Array(Float64, 2)
FiniteDiff.hessian!(tmp4, x -> sin(x[1]) + cos(x[2]), [1.0, 1.0], buffer)
tmp4, [-sin(1.0) 0.0; 0.0 -cos(1.0)]
```

Finally, let's look at the higher-order function API:

```jl
import FiniteDiff

# Compute approximate derivative and compare it with the true value.
f′ = FiniteDiff.derivative(sin)
f′(0.0), cos(0.0)

# Compute approximate second derivative and compare it with the true value.
f′′ = FiniteDiff.second_derivative(sin)
f′′(1.0), -sin(1.0)

# Compute approximate gradient and compare it with the true value.
g = FiniteDiff.gradient(x -> sin(x[1]))
g([0.0]), [cos(0.0)]

# Compare approximate Hessian and compare it with the true value.
h = FiniteDiff.hessian(x -> sin(x[1]) + cos(x[2]))
h([1.0, 1.0]), [-sin(1.0) 0.0; 0.0 -cos(1.0)]
```
