module Rolling
import ShiftedArrays.lag, ShiftedArrays.lead
import MaxMinFilters.movmaxmin
using ..utils

export rolling_sum_1d, rolling_prod_1d, rolling_func_1d, rolling_norm_1d

function rolling_sum_1d(arr, window, default=NaN)
    """ Rolling sum over a 1d array """
    len = length(arr)
    rsum = typeof(arr)(undef, len)
    rsum[1:window] .= default
    rol = sum(view(arr, 1:window))
    rsum[window] = rol
    for n in window + 1:len
        rol = rol - arr[n - window] + arr[n]
        rsum[n] = rol
    end
    return rsum
end

function rolling_prod_1d(arr, window, default=NaN)
    len = length(arr)
    res = typeof(arr)(undef, len)
    res[1:window] .= default
    acc = prod(view(arr, 1:window))
    res[window] = acc
    for n in window + 1:len
        acc = acc / arr[n - window] * arr[n]
        res[n] = acc
    end
    # the logsum/exp is 4x slower
    return res
end

function rolling_func_1d(arr, split, apply, combine=(x) -> x;
                                             window, acc=0, default=NaN)
    len = length(arr)
    res = typeof(arr)(undef, len)
    res[1:window - 1] .= default
    for n in 1:window
        acc = apply(acc, arr[n])
    end
    res[window] = combine(acc)
    @inbounds for n in window + 1:len
        acc = apply(
                split(acc, arr[n - window]),
                arr[n])
        res[n] = combine(acc)
    end
    return res
end


function rolling_norm_1d(arr, window)
    mx, mn = movmaxmin(arr, window)
    return (arr .- mn ) ./ (mx .- mn)
end

end
