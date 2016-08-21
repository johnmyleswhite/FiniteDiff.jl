immutable HessianFunction{T <: Function}
    f::T
    name::String
end

hess_funcs = (
    HessianFunction(
        FiniteDiff.hessian,
        "FiniteDiff.hessian(f, x)",
    ),
    HessianFunction(
        Calculus.finite_difference_hessian,
        "Calculus.finite_difference_hessian(f, x)",
    ),
)
