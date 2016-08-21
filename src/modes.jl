"""
    Mode

# Description

A type that represents the mode of finite differences being used.

* Super type: `Any`
"""
abstract Mode

"""
    ForwardMode

# Description

A type that represents the use of forward mode finite differences for
gradients.

* Super type: `Mode`
"""
immutable ForwardMode <: Mode
end

"""
    BackwardMode

# Description

A type that represents the use of backward mode finite differences for
gradients.

* Super type: `Mode`
"""
immutable BackwardMode <: Mode
end

"""
    CentralMode

# Description

A type that represents the use of central mode finite differences for
gradients.

* Super type: `Mode`
"""
immutable CentralMode <: Mode
end

"""
    ComplexMode

# Description

A type that represents the use of complex mode finite differences for
gradients.

* Super type: `Mode`
"""
immutable ComplexMode <: Mode
end

"""
    HessianMode

# Description

A type that represents the use of finite differences for hessians.

* Super type: `Mode`
"""
immutable HessianMode <: Mode
end
