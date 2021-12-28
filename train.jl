include("config.jl")
include("markov_game.jl")
include("model.jl")
include("logger.jl")
using Dates: now
using BSON: @save

function train(cf::Config)
    model_predator = get_model(cf)
    log_info(log_path, "Initialized Predator's model")
    model_prey = get_model(cf)
    log_info(log_path, "Initialized Prey's model")

    losses = (
        predator = Dict{Tuple{Int64, Int64}, Float64}(),
        prey = Dict{Tuple{Int64, Int64}, Float64}(),
    )
    rewards = (
        predator = Dict{Tuple{Int64, Int64}, Float64}(),
        prey = Dict{Tuple{Int64, Int64}, Float64}(),
    ) 

    for iter in 1 : cf.num_iterations
        s = get_random_state(cf)
        for step in 1 : cf.num_steps
            # Transform the current state to observations for each agent
            # Each agent chooses an action with Softmax-response strategy
            ob_predators = [get_observation(s, p) for p in s.predators]
            a_predators = [softmax_response(model_predator, ob) for ob in ob_predators]
            ob_preys = [get_observation(s, p) for p in s.preys]
            a_preys = [softmax_response(model_prey, ob) for ob in ob_preys]

            # Forward from the current state to get next state and reward values for each agent
            s_next, rw_predators, rw_preys = forward(s, a_predators, a_preys)

            # Calculate the "utility" values of the next state
            u_predators = get_utility(s_next, model_predator, s_next.predators)
            u_preys = get_utility(s_next, model_prey, s_next.preys)

            # Optimize model weights to fit the actual rewards
            loss_predators, loss_preys = [], []
            γ = 0.9
            for (ob, a, rw, u) in zip(ob_predators, a_predators, rw_predators, u_predators)
                push!(loss_predators, train_step(model_predator, ob, a, rw + γ*u))
            end
            for (ob, a, rw, u) in zip(ob_preys, a_preys, rw_preys, u_preys)
                push!(loss_preys, train_step(model_prey, ob, a, rw + γ*u))
            end

            losses.predator[(iter, step)] = mean(loss_predators)
            losses.prey[(iter, step)] = mean(loss_preys)
            rewards.predator[(iter, step)] = mean(rw_predators)
            rewards.prey[(iter, step)] = mean(rw_preys)
            log_train_step(log_path, iter, step, losses, rewards)
            s = s_next
        end
    end

    @save model_path*"predator.bson" model_predator
    @save model_path*"prey.bson" model_prey
    @save log_path*"train_losses.bson" losses
    @save log_path*"train_rewards.bson" rewards
end

cf = get_config_1()
log_path = "./log/" * string(format(now(), "YYYY-mm-dd_HH:MM:SS")) * "/"
model_path = log_path * "/model/"
mkpath(log_path)
mkpath(model_path)
log_config(log_path, cf)
train(cf)