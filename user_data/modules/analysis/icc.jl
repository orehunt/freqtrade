module ICCorr
using ..utils
using StatsBase
import Base.@propagate_inbounds

export icc

@propagate_inbounds function icc(data::AbstractArray, targets::AbstractArray, norm_ratio::Bool=true)
    """
    - a group is composed by 1 column from data, and all columns from targets
    - every row is a an observation
    - the sample size is the number of rows, which must match between data targets
    - the number of columns in data determins the number of tests

    K is the number of elements in each group, targets columns + 1
    N is the number of samples over which mean is applied (rows)
    C is the number of tests to compute, data columns, output will have size C
    """
    sz = size(data)
    N, C = sz[1], sz[2]

    if ndims(targets) == 1
        targets = reshape(targets, (N, 1))
        K = 2
    else
        K = size(targets)[2] + 1
    end
    @assert N == size(targets)[1]

    targets = copy(targets); targets[isnan.(targets)] .= 0
    data = copy(data); data[isnan.(data)] .= 0
    # since the targets are shared for all groups
    # compute the sum of for every sample
    targets_sum = sum(targets, dims=2)
    @assert length(targets_sum) == N
    # the size of each test, num samples * group size
    T = N * K

    # s^2
    s2 = @newarr(C); sigma = @newarr(C)
    # pre alloc
    group_sum_1d = @newarr(N)
    @inbounds for c in 1:C
        samples = view(data, :, c)
        group_sum_1d[:] = samples .+ targets_sum
        # the total mean of the test
        # mu_t = @skipnan(sum, group_sum_1d) / T
        mu_t = sum(group_sum_1d) / T
        # calc s2 for this test, compute (x@n - mu_t)^2 for every
        # member of the group (1 + targets_cols), then reduce all the samples
        s2[c] = (sum((samples .- mu_t).^2) +
        #  all the groups share these members
                 sum((targets .- mu_t).^2)) / T
        # the mean of the group for each sample for each test (row mean)
        # minus the total mean, squared, then reduce by sum
        sigma[c] = sum((group_sum_1d ./ K .- mu_t).^2)
    end
    # ratio
    r = K / (K - 1) .* (sigma ./ N) ./ s2 .- (1 / (K - 1))
    @assert length(r) == C

    if norm_ratio
        @unit_range!(r, norm_ratio)
    end
    return r
end

end
