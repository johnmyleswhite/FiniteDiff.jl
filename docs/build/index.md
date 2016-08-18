
<a id='FiniteDiff.jl-1'></a>

# FiniteDiff.jl


<a id='Introduction-1'></a>

## Introduction


The FiniteDiff.jl package provides functions for computing approximate first and second derivatives of univariate and multivariate functions. It uses finite-differencing to generate these approximations, which means that it is possible to compute an approximate derivative for any Julia function – even functions that are not differentiable and should not be differentiated.


Despite being universally applicable, the quality of the approximated derivatives computed by this package varies substantially across different functions. Even for a fixed function, the quality of this approximation varies substantially across different points in the function's domain. If your function is amenable to other techniques (such as forward-mode automatic differentiation), you will likely get better results from those techniques. Use finite-differencing only when you need approximate derivatives and no other technique will work. (An example of this is when you are calling out to a C function, because the function is not written in pure Julia, it will not be possible to use automatic differentiation to determine its derivatives.)


This package is intended to be a replacement for the finite-differencing functionality found in the Calculus package. It attempts to unify the API for finite-differencing with the API for automatic differentiation found in the [ForwardDiff.jl package](http://www.juliadiff.org/ForwardDiff.jl/index.html).


<a id='Core-API-1'></a>

## Core API


Most users will want to work with the following core functions:


  * `derivative()`: Use this for functions from $\mathbb{R}$ to $\mathbb{R}$.
  * `second_derivative()`: Use this for functions from $\mathbb{R} to $\mathbb{R}$.
  * `gradient()`: Use this for functions from $\mathbb{R^n}$ to $\mathbb{R}$.
  * `hessian()`: Use this for functions from $\mathbb{R^n}$ to $\mathbb{R}$.


All of these functions also come in mutating variants, which can offer substantially better performance in many settings by avoiding the creation of temporaries. In addition, these functions come in higher-order variants, which allow you to create a new function that will generate the output of these functions at any input you provide.


  * Pure functions
  * Mutating functions
  * Higher-order functions


<a id='Modes-1'></a>

# Modes


<a id='Step-Sizes-1'></a>

# Step Sizes

