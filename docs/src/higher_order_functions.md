# Derivative

# Second Derivative

# Gradient

# Hessian

```jl
import FiniteDiff

# Compute an approximate derivative and compare it with the true value.
f′ = FiniteDiff.derivative(sin)
f′(0.0), cos(0.0)

# Compute an approximate second derivative and compare it with the true value.
f′′ = FiniteDiff.second_derivative(sin)
f′′(1.0), -sin(1.0)

# Compute an approximate gradient and compare it with the true value.
g = FiniteDiff.gradient(x -> sin(x[1]))
g([0.0]), [cos(0.0)]

# Compare an approximate Hessian and compare it with the true value.
h = FiniteDiff.hessian(x -> sin(x[1]) + cos(x[2]))
h([1.0, 1.0]), [-sin(1.0) 0.0; 0.0 -cos(1.0)]
```
