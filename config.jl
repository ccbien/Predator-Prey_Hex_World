using Parameters

@with_kw struct Config
    n_rows::Int32
    n_cols::Int32
    num_predators::Int32
    num_preys::Int32
    ε::Float32 # ε-greedy
    
    epochs::Int32
    steps::Int32
end

function get_config_1()::Config
    kwargs = Dict(
        :n_rows => 5,
        :n_cols => 5,
        :num_predators => 2,
        :num_preys => 2,
        :ε => 0.1,
        :epochs => 10,
        :steps => 1000
    )
    return Config(;kwargs...)
end