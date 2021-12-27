using Parameters: @with_kw

@with_kw struct Config
    n_rows::Int64
    n_cols::Int64
    num_predators::Int64
    num_preys::Int64

    hidden_layers::Vector{Int64}
    num_iterations::Int64
    num_steps::Int64
end

function get_config_1()::Config
    kwargs = Dict(
        :n_rows => 4,
        :n_cols => 8,
        :num_predators => 5,
        :num_preys => 5,
        :hidden_layers => [32, 16],
        :num_iterations => 100,
        :num_steps => 100
    )
    return Config(;kwargs...)
end

function get_config_2()::Config
    kwargs = Dict(
        :n_rows => 4,
        :n_cols => 8,
        :num_predators => 2,
        :num_preys => 2,
        :hidden_layers => [32, 16],
        :num_iterations => 100,
        :num_steps => 100
    )
    return Config(;kwargs...)
end