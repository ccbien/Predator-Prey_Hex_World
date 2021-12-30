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

function get_config(name::String)::Config
    kwargs = Dict()
    if name == "1"
        kwargs = Dict(
            :n_rows => 4,
            :n_cols => 8,
            :num_predators => 1,
            :num_preys => 1,
            :hidden_layers => [16, 8],
            :num_iterations => 100,
            :num_steps => 100,
            :η => 0.01,
            :β => (0.9, 0.999),
        )
    elseif name == "2"
        kwargs = Dict(
            :n_rows => 5,
            :n_cols => 10,
            :num_predators => 3,
            :num_preys => 3,
            :hidden_layers => [32, 16],
            :num_iterations => 1000,
            :num_steps => 1000,
            :η => 0.01,
            :β => (0.9, 0.999),
        )
    end
    return Config(;kwargs...)
end
