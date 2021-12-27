function get_vector(r0::Int64, c0::Int64, r1::Int64, c1::Int64)::Tuple{Float64, Float64}
    return (r1 - r0, (c1 - c0) * sqrt(3) / 4)
end