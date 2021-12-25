include("config.jl")

struct State
    n_rows::Integer
    n_cols::Integer
    predators::Array{Tuple{Int32, Int32}}
    preys::Array{Tuple{Int32, Int32}}
end

function get_random_state(config::Config)
end

function transition(s, a)
end