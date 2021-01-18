module utils
using IterTools
using StatsBase

export @unit_range!, @skipnan, @swapnan, filsoc, @newarr, @passnan, @filtnan, unit_range, @fltinf, @swapinf, @arrfloat

macro newarr(dims, type=Float64)
    quote
        Array{$(esc(type))}(undef, $(esc(dims)))
    end
end

macro passinf(arr, val)
    quote
        imap((el) -> isfinite(el) ? el : $(esc(val)), $(esc(arr)))
    end
end

macro passnan(arr, val)
    quote
        imap((el) -> isnan(el) ? $(esc(val)) : el, $(esc(arr)))
    end
end

macro arrfloat(arr, yes=true)
    quote
        arr = $(esc(arr))
        if $yes == true
            Array{Float64,ndims(arr)}(arr)
        else
            arr
        end
    end
end

macro swapinf(arr, conv=false, nanv=0, infv=1)
    quote
        @arrfloat(map((el) -> isfinite(el) ? el :
            (isnan(el) ? $(esc(nanv)) :
             sign(el) * $(esc(infv))), $(esc(arr))), $conv)
    end
end

macro swapnan(arr, val)
    quote
        map((el) -> isnan(el) ? $(esc(val)) : el, $(esc(arr)))
    end
end

macro filtnan(arr)
    quote
        filter(!isnan, $(esc(arr)))
    end
end

macro skipnan(f, arr, dims=nothing)
    if isnothing(dims)
        quote
            $f(filter(!isnan, $(esc(arr))))
        end
    else
        quote
            mapslices(x -> $f(filter(!isnan, x)),
                        $(esc(arr)), dims=$dims)
        end
    end
end

macro fltinf(arr)
    quote
        filter(isfinite, $(esc(arr)))
    end
end

macro _maparr(f, arr, dims, pred=!isnan)
    if isnothing(dims)
        quote
            $f(filter($pred, $(esc(arr))))
        end
    else
        quote
            mapslices(x -> $f(filter($pred, x)),
                        $(esc(arr)), dims=$dims)
        end
    end
end

function unit_range(arr)
    return StatsBase.transform(fit(UnitRangeTransform, arr), arr)
end

macro unit_range!(arr, yes=true)
    if yes == true
        quote
            arr = $(esc(arr))
            return StatsBase.transform!(fit(UnitRangeTransform, arr), arr)
        end
    end
end

function filsoc(arr, pct, match; inv::Bool=false, concat::Bool=true)
    """ Filter an array above (below) a value, sort and concat the result of
    another array of same input length at the equivalent sorted index
         """
    pct_mask = inv ? arr .< pct : arr .> pct
    sort_mask = sortperm(arr[pct_mask, :])
    values = arr[pct_mask, :][sort_mask, :]

    if concat & (match !== nothing)
        values = hcat(values, match[pct_mask, :][sort_mask, :])
    end
    return values
end

function unzip(a)
    return map(x -> getfield.(a, x), fieldnames(eltype(a)))
end

end
