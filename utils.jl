function get_vector(r0::Int64, c0::Int64, r1::Int64, c1::Int64)::Tuple{Float64, Float64}
    return (r1 - r0, (c1 - c0) * sqrt(3) / 4)
end

function is_outside(s::State, r, c)
    return r < 1 || s.n_rows < r || c < 1 || s.n_cols < c
end

function get_count_dict(items::Vector{Tuple{Int64, Int64}})::Dict{Tuple{Int64, Int64}, Int64}
    d = Dict()
    for item in items
        d[item] = get(d, item, 0) + 1
    end
    return d
end

function get_distance_to_the_nearest(r0::Int64, c0::Int64, items::Vector{Tuple{Int64, Int64}})
    dist = 10^9
    for (r, c) in items
        d = abs(r0 - r)
        left, right = c0 - d, c0 + d
        if c < left
            d += Int64((left - c) / 2)
        elseif c > right
            d += Int64((c - right) / 2)
        end
        dist = min(dist, d)
    end
    return dist
end