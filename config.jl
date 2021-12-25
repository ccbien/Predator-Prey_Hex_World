struct Config
    n_rows::Int32
    n_cols::Int32
    num_predators::Int32
    num_preys::Int32
    obstacles::Array{Tuple{Int32, Int32}}
end

function get_config_1()::Config
    args = [3, 10, 2, 2, [(1, 9), (2, 2), (2, 6), (3, 9)]]
    return Config(args...)
end