include("config.jl")
include("markov_game.jl")
include("model.jl")
include("logger.jl")
using Dates: now
using BSON: @save

function train(cf::Config)
    model_predator = get_model(cf)
    model_prey = get_model(cf)

    for iter in 1 : cf.num_iterations
        s = get_random_state(cf)
        for step in 1 : cf.num_steps
            # Transform the current state to observations for each agent
            # Each agent chooses an action with Softmax-response strategy
            ob_predators = get_observation.(s, s.predators)
            a_predators = [softmax_response(model_predator, ob) for ob in ob_predators]
            ob_preys = get_observation.(s, s.preys)
            a_preys = [softmax_response(model_prey, ob) for ob in ob_preys]

            # Forward from the current state to get next state and reward values for each agent
            s_next, rw_predators, rw_preys = forward(s, a_predators, a_preys)

            # Optimize model weights to fit the actual rewards
            for (ob, a, rw) in zip(ob_predators, a_predators, rw_predators)
                loss = train_step(model_predator, ob, a, rw)
            end
            for (ob, a, rw) in zip(ob_preys, a_preys, rw_preys)
                loss = train_step(model_prey, ob, a, rw)
            end

            s = s_next
        end
    end

    @save model_path*"predator.bson" model_predator
    @save model_path*"prey.bson" model_prey
end

cf = get_config_1()
log_path = "./log/" * string(now()) * "/"
model_path = log_path * "/model/"
mkpath(log_path)
mkpath(model_path)
log_config(log_path, cf)
train(cf)