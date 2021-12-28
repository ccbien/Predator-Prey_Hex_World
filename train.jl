include("config.jl")
include("markov_game.jl")
include("model.jl")
include("logger.jl")
using Dates: now
using BSON: @save

function get_actions(obs::Vector{Matrix{Float64}})::Vector{Tuple{Int64, Int64}}
    a = []
    for ob in obs
        push!(a, softmax_response(model_predator, ob))
    return a

function train(cf::Config)
    model_predator = get_nn(cf)
    model_prey = get_nn(cf)

    for iter in 1 : cf.num_iterations
        s = get_random_state(cf)
        for step in 1 : cf.num_steps
            ob_predators = get_observation.(s, s.predators)
            a_predators = get_actions(ob_predators)

            ob_preys = get_observation.(s, s.preys)
            a_preys = get_actions(ob_preys)

            s_next, rw_predators, rw_preys = forward(s, a_predators, a_preys)
        end
    end

    @save log_path * "predator.bson" model_predator
    @save log_path * "prey.bson" model_prey
end

cf = get_config_1()
log_path = "./log/" * string(now()) * "/"
mkpath(log_path)
log_config(log_path, cf)
train(cf)