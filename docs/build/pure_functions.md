
<a id='Derivative-1'></a>

# Derivative


<a id='Second-Derivative-1'></a>

# Second Derivative


<a id='Gradient-1'></a>

# Gradient


<a id='Hessian-1'></a>

# Hessian


```
derivative(f::Function, x::AbstractFloat, ::ForwardMode)
```


```
derivative(f::Function, x::AbstractFloat, ::CentralMode)
```


```jl
import FiniteDiff

# Compute an approximate derivative and compare it with the true value.
d = FiniteDiff.derivative(sin, 0.0)
d, cos(0.0)

# Compute an approximate second derivative and compare it with the true value.
d2 = FiniteDiff.second_derivative(sin, 1.0)
d2, -sin(1.0)

# Compute an approximate gradient and compare it with the true value.
gr = FiniteDiff.gradient(x -> sin(x[1]), [0.0])
gr, [cos(0.0)]

# Compare an approximate Hessian and compare it with the true value.
H = FiniteDiff.hessian(x -> sin(x[1]) + cos(x[2]), [1.0, 1.0])
H, [-sin(1.0) 0.0; 0.0 -cos(1.0)]
```

