include("config.jl")
include("markov_game.jl")
include("model.jl")
include("logger.jl")
using Dates: now
using BSON: @save

function train(cf::Config)
    model_predator = get_nn(cf)
    model_prey = get_nn(cf)

    for iter in 1 : cf.num_iterations
        s = get_random_state(cf)
        for step in 1 : cf.num_steps
            ob_predators = get_observation.(s, s.predators)
            ob_preys = get_observation.(s, s.preys)
            
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