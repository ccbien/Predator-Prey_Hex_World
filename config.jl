using Parameters: @with_kw

@with_kw struct Config
    n_rows::Int32
    n_cols::Int32
    num_predators::Int32
    num_preys::Int32
    num_iterations::Int32
    num_steps::Int32
end

function get_config_1()::Config
    kwargs = Dict(
        :n_rows => 4,
        :n_cols => 8,
        :num_predators => 1,
        :num_preys => 1,
        :num_iterations => 100,
        :num_steps => 100
    )
    return Config(;kwargs...)
end