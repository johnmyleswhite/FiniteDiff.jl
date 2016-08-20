# Introduction

The FiniteDiff.jl package provides functions for computing approximate first
and second derivatives of univariate and multivariate functions. It uses
numerical differentiation to generate these approximations, which means that it
is possible to compute an approximate derivative for any Julia function -- even
functions that are not differentiable and are therefore not safe to
differentiate.

Despite being universally applicable, the quality of the approximated
derivatives computed by this package varies substantially across different
functions. Even for a fixed function, the quality of the approximations
provided by numerical differentiation varies substantially across different
points in the function's domain. If your function is amenable to other
techniques (such as forward-mode automatic differentiation), you will almost
certainly get better results from those techniques. Use this package only when
you need approximate derivatives and no other technique will work.

An example of a setting in which you would need to use numerical
differentiation involves a Julia function that calls out to a C function to
compute a mathematical function along the way. This might happen if you were
to use a C implementation to compute the integral of the incomplete beta
function as part of a statistical calculation. Because the C function is not
written in pure Julia, automatic differentiation techniques would not work on
the Julia code as a whole. In that kind of setting, numerical differentiation
would be the only effective way to automatically determine the Julia function's
derivatives.

This package is intended to be a replacement for the numerical differentiation
functionality found in the Calculus package. It attempts to unify the API
for numerical differentiation with the API for automatic differentiation found
in the
[ForwardDiff.jl package](http://www.juliadiff.org/ForwardDiff.jl/index.html).

# Basic Examples

The examples below are intended to provide a basic sense of how the package
can be used. Further documentation is found in others sections.

```julia
julia> import FiniteDiff: derivative, second_derivative, gradient, hessian

julia> y′ = derivative(sin, 1.0)
0.5403023058631036

julia> y′′ = second_derivative(sin, 1.0)
-0.8414709866046906

julia> gr = gradient(x -> (1.0 - x[1])^2 + (2.0 - x[2])^2, [0.0, 0.0])
2-element Array{Float64,1}:
 -2.0
 -4.0

julia> H = hessian(x -> (1.0 - x[1])^2 + (2.0 - x[2])^2, [0.0, 0.0])
2×2 Array{Float64,2}:
 2.0         6.05545e-6
 6.05545e-6  2.0
```

The four functions shown above define the core API for the FiniteDiff package.
Each of the functions has a section dedicated to it; that function's section
provides a full overview of the function's use.

Before you read the detailed documentation, we encourage you to consider using
mutating variants of the `gradient` and `hessian` functions to avoid allocating
memory unnecessarily when working with multivariate functions. The examples
below show how these mutating variants work:

```julia
julia> import FiniteDiff: gradient!, hessian!

# Allocate memory once that can be reused in the future.
julia> gr = Array(Float64, 2)
2-element Array{Float64,1}:
 1.4822e-323
 4.44659e-323

julia> H = Array(Float64, 2, 2)
2×2 Array{Float64,2}:
 1.50626e165  6.97302e252
 6.46167e174  8.40996e-315

julia> buffer = Array(Float64, 2)
2-element Array{Float64,1}:
 2.27838e-314
 2.25293e-314

julia> x = [0.0, 0.0]
2-element Array{Float64,1}:
 0.0
 0.0

julia> gradient!(gr, x -> (1.0 - x[1])^2 + (2.0 - x[2])^2, x, buffer)

# The gradient is now stored in gr.
julia> gr
2-element Array{Float64,1}:
 -2.0
 -4.0

julia> hessian!(H, x -> (1.0 - x[1])^2 + (2.0 - x[2])^2, x, buffer)

# The Hessian is now stored in H.
julia> H
2×2 Array{Float64,2}:
 2.0         6.05545e-6
 6.05545e-6  2.0
```

Note that these functions require an additional buffer of memory to operate on.
This buffer will be mutated, but its output at the end of the function's
execution is essentially meaningless. It is just a scratchpad used to speed up
the computation of the numerical approximations.
