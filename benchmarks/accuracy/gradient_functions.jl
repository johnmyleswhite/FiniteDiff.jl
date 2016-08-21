immutable GradientFunction{T <: Function}
    f::T
    name::String
end

grad_funcs = (
    GradientFunction(
        FiniteDiff.gradient,
        "FiniteDiff.gradient(f, x)",
    ),
    GradientFunction(
        (f, x) -> FiniteDiff.gradient(f, x, FiniteDiff.ForwardMode()),
        "FiniteDiff.gradient(f, x, FiniteDiff.ForwardMode())",
    ),
    GradientFunction(
        (f, x) -> FiniteDiff.gradient(f, x, FiniteDiff.BackwardMode()),
        "FiniteDiff.gradient(f, x, FiniteDiff.BackwardMode())",
    ),
    GradientFunction(
        (f, x) -> FiniteDiff.gradient(f, x, FiniteDiff.CentralMode()),
        "FiniteDiff.gradient(f, x, FiniteDiff.CentralMode())",
    ),
    GradientFunction(
        (f, x) -> FiniteDiff.gradient(f, x, FiniteDiff.ComplexMode()),
        "FiniteDiff.gradient(f, x, FiniteDiff.ComplexMode())",
    ),
    GradientFunction(
        Calculus.finite_difference,
        "Calculus.finite_difference(f, x)",
    ),
)
