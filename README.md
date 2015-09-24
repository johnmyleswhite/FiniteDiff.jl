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

    using FiniteDiff

    # Compare with cos(0.0)
    FiniteDiff.derivative(sin, 0.0)

    # Compare with [cos(0.0), -sin(0.0)]
    FiniteDiff.gradient(x -> sin(x[1]) + cos(x[2]), [0.0, 0.0])

    # Compare with -sin(1.0)
    FiniteDiff.second_derivative(sin, 1.0)

    # Compare with [-sin(1.0) 0.0; 0.0 -cos(1.0)]
    FiniteDiff.hessian(x -> sin(x[1]) + cos(x[2]), [1.0, 1.0])
