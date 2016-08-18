
<a id='Derivative-1'></a>

# Derivative


<a id='Second-Derivative-1'></a>

# Second Derivative


<a id='Gradient-1'></a>

# Gradient


<a id='Hessian-1'></a>

# Hessian


```jl
import FiniteDiff

# Compute an approximate derivative and compare it with the true value.
tmp1 = Array(Float64, 1)
FiniteDiff.derivative!(tmp1, sin, 0.0)
tmp1, [cos(0.0)]

# Compute an approximate second derivative and compare it with the true value
tmp2 = Array(Float64, 1)
FiniteDiff.second_derivative!(tmp2, sin, 1.0)
tmp2, [-sin(1.0)]

# Compute an approximate gradient and compare it with true value
tmp3 = Array(Float64, 1)
buffer = Array(Float64, 1)
FiniteDiff.gradient!(tmp3, x -> sin(x[1]), [0.0], buffer)
tmp3, [cos(0.0)]

# Compare an approximate Hessian and compare it with the true value.
tmp4 = Array(Float64, 2, 2)
buffer = Array(Float64, 2)
FiniteDiff.hessian!(tmp4, x -> sin(x[1]) + cos(x[2]), [1.0, 1.0], buffer)
tmp4, [-sin(1.0) 0.0; 0.0 -cos(1.0)]
```

