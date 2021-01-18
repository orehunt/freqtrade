module Liquidity

using ..Rolling
using ..utils
import ShiftedArrays.lag
import ShiftedArrays.lead
import StatsBase.mean

export illiquidity, liquidity, calc_spread, sim_high_low, calc_skewed_spread, skew_rate_by_liq

# amihud illiquidity measure
@inline function illiquidity(close, volume, window=120)
    # volume in quote currency
    volume_curr = volume .* close
    # returns are NOT a ratio
    returns_volume_ratio = abs.(@passnan((close .- lag(close, 1, default=close[1])) ./
                 volume_curr, 0))
    rolling_rvr_sum = rolling_sum_1d(returns_volume_ratio, window)
    return rolling_rvr_sum ./ window .* 1e6
end

# LIX formula
# values between ~5..~10 higher is more liquid
@inline function liquidity(volume, close, high, low)
    return log10.((volume .* close) ./ (high .- low))
end

@inline function sim_high_low(open, close)
    return close .<= open
end


function calc_spread(high, low, close)
    """ A Simple Estimation of Bid Ask spread
    NOTE: this calculation can return NaNs """
    # calc mid price
    mid_range = (log.(high) .+ log.(low)) ./ 2
    # # forward mid price
    mid_range_1 = (log.(lead(high, default=NaN)) .+ log.(lead(low, default=NaN))) ./ 2
    log_close = log.(close)

    # # spread formula
    return sqrt.(max.(4 .* (log_close .- mid_range) .* (log_close .- mid_range_1), 0))
end

function calc_skewed_spread(high, low, close, volume, wnd)
    """ Calculate spread and skew it according to the balance of LIX and illiquidity estimates """
    spread = calc_spread(high, low, close)
    # min max liquidity statistics over a rolling window to use for interpolation
    lix = liquidity(volume, close, high, low)
    lix_norm = rolling_norm_1d(@swapinf(lix, conv = true), wnd)
    ilq = illiquidity(close, volume, wnd)
    ilq_norm = rolling_norm_1d(@swapinf(ilq, conv = true), wnd)
    # skew by liquidity indicator depending on which one is dominant;
    # if illiquidity > liquidity, spread will be higher, therefore
    # alpha is the illiquidity (alpha increases the values, beta lowers),
    # however since we are skewing the spread don't take values lower than 0
    skewed = skew_by_ratio(alpha=ilq_norm, beta=lix_norm, rate=spread)
    return max.(0, skewed)
end

function skew_by_ratio(;alpha, beta, rate, ranged=true)
    if !ranged
        alpha = unit_range(alpha)
        beta = unit_range(beta)
    end
    # alpha/beta
    ratio = alpha ./ (alpha + beta)
    # alpha increases the rate
    alpha_w = (alpha + ratio .* alpha)
    # beta lowers the rate
    beta_w = (beta + (1 .- ratio) .* beta)
    rate[:] = (alpha_w - beta_w) .* rate + rate
    return rate
end

end
