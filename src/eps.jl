macro forward(x)
    x = esc(x)
    quote
        sqrt(eps(eltype($x))) * max(one(eltype($x)), abs($x))
    end
end

macro backward(x)
    x = esc(x)
    quote
        sqrt(eps(eltype($x))) * max(one(eltype($x)), abs($x))
    end
end

macro central(x)
    x = esc(x)
    quote
        cbrt(eps(eltype($x))) * max(one(eltype($x)), abs($x))
    end
end

macro complex(x)
    x = esc(x)
    quote
        eps($x)
    end
end

macro hessian(x)
    x = esc(x)
    quote
        eps(eltype($x))^(1/4) * max(one(eltype($x)), abs($x))
    end
end
