include("config.jl")
include("markov_game.jl")
include("model.jl")
include("logger.jl")
using BSON: @save

function train(cf::Config)
    logging_interval = 1
    if length(ARGS) > 1 logging_interval = parse(Int64, ARGS[2]) end

    model_predator = get_model(cf)
    model_prey = get_model(cf)
    
    log_train_info(log_path, "Start training")
    for iter in 1 : cf.num_iterations
        s = get_random_state(cf)
        losses = (predator = [], prey = [])
        rewards = (predator = [], prey = [])

        for step in 1 : cf.num_steps
            # Transform the current state to observations for each agent
            # Each agent chooses an action with Softmax-response strategy
            ob_predators = [get_observation(s, p) for p in s.predators]
            a_predators = [softmax_response(model_predator, ob) for ob in ob_predators]
            ob_preys = [get_observation(s, p) for p in s.preys]
            a_preys = [softmax_response(model_prey, ob) for ob in ob_preys]

            # Forward from the current state to get next state and reward values for each agent
            s_next, rw_predators, rw_preys, _, _ = forward(s, a_predators, a_preys)

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

            push!(losses.predator, mean(loss_predators))
            push!(losses.prey, mean(loss_preys))
            push!(rewards.predator, mean(rw_predators))
            push!(rewards.prey, mean(rw_preys))
            
            s = s_next
        end

        @save model_path*"predator.bson" model_predator
        @save model_path*"prey.bson" model_prey
        @save log_path*"train_losses.bson" losses
        @save log_path*"train_rewards.bson" rewards

        if iter % logging_interval == 0
             log_train_iteration(log_path, iter, cf.num_steps, losses, rewards)
        end
    end

end

println("Train on config " * ARGS[1])
cf = get_config(ARGS[1])
log_path = "./log/" * ARGS[1] * "/"
model_path = log_path * "/model/"

rm(log_path; force=true, recursive=true)
mkpath(log_path)
mkpath(model_path)
log_config(log_path, cf)
train(cf)