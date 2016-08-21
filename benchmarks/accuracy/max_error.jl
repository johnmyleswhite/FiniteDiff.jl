function max_error(g_true, g_approx)
    n = length(g_true)
    @assert length(g_approx) == n
    err = 0.0
    for i in 1:n
        abs_err = g_true[i] - g_approx[i]
        err = max(err, abs(abs_err))
    end
    return err
end
