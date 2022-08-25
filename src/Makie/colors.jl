# [max val, growth rate, xshift, yshift]
# @. sigmoid(x, ps::Vector) = ps[1] / (1 + exp(ps[2] * (x - ps[3]))) + ps[4]
@. sigmoid(x, p1 = 1.0, p2 = -1.0, p3 = 0.0, p4 = 0.0) = p1 / (1 + exp(p2 * (x - p3))) + p4
function _sigmoid_range(x0, x1; length = 10)
    if length > 0
        vals = sigmoid(range(-5, 5, length=length), 1.0, -1.0, 0.0, 0.0)
        vals ./= vals[end]
        return (x1 - x0) .* vals .+ x0
    else
        return Float64[]
    end
end

CBKRY(args...; kwargs...) = APS_CBKRY(args...; kwargs...)
function APS_CBKRY(N = 200; include_white = false, power = 2, sharp = false)
    n = max(1, div(N, 4+include_white))
    n1 = include_white ? n+1 : 0
    
    f = sharp ? _sigmoid_range : range
    
    hues1 = vcat(f(60, 50, length = n1), f(50, 0, length=n+1)[1+include_white:end], f(0, 0, length=n))
    sats1 = [1 - ((3n - x) / 3n)^power for x in n*(1-include_white):3n]
    vals1 = vcat(ones(n1), f(1, 0.7, length=n+1)[1+include_white:end], f(0.7, 0, length=n+1)[2:end])
    cs1 = HSV.(hues1, sats1, vals1)

    hues2 = vcat(f(240, 240, length = n+1)[2:end], f(240, 200, length=n+1)[2:end])
    sats2 = [1 - (x / 3n)^power for x in 0:2n-1]
    vals2 = vcat(f(0, 0.7, length = n+1)[2:end], f(0.7, 1, length=n+1)[2:end])
    cs2 = HSV.(hues2, sats2, vals2)

    return RGBf.(vcat(cs1, cs2))
end