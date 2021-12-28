using Parameters: @with_kw

@with_kw struct Config
    n_rows::Int64
    n_cols::Int64
    num_predators::Int64
    num_preys::Int64

    hidden_layers::Vector{Int64}
    num_iterations::Int64
    num_steps::Int64

    # Adam optimizer
    η::Float64
    β::Tuple{Float64, Float64}
end

function get_config_1()::Config
    kwargs = Dict(
        :n_rows => 4,
        :n_cols => 8,
        :num_predators => 3,
        :num_preys => 3,
        :hidden_layers => [32, 16],
        :num_iterations => 10,
        :num_steps => 10,
        :η => 0.001,
        :β => (0.9, 0.999),
    )
    return Config(;kwargs...)
end