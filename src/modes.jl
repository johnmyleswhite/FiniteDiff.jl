"""
A type that represents the mode of finite-differencing being used.
"""
abstract Mode

"""
A type that represents the use of forward finite-differencing for gradients.
"""
immutable ForwardMode <: Mode
end

"""
A type that represents the use of backward finite-differencing for gradients.
"""
immutable BackwardMode <: Mode
end

"""
A type that represents the use of central finite-differencing for gradients.
"""
immutable CentralMode <: Mode
end

"""
A type that represents the use of complex finite-differencing for gradients.
"""
immutable ComplexMode <: Mode
end

"""
A type that represents the use of finite-differencing for hessians.
"""
immutable HessianMode <: Mode
end
