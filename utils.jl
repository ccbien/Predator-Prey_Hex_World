function get_coordinate(r, c)::Tuple{Float64, Float64}
    return (r - 1, (c - 1) * sqrt(3) / 4)
end