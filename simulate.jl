include("config.jl")
include("markov_game.jl")
include("model.jl")
include("logger.jl")
using Flux
using BSON: @save, @load

function simulate(cf::Config, res::NamedTuple, name::String)
    @load model_path*"predator.bson" model_predator
    @load model_path*"prey.bson" model_prey


    log_prefix = log_path * "sim_" * name * "_" * string(num_steps)
    s = get_random_state(cf)
    
    predator_stats = [PredatorStat() for i in 1 : cf.num_predators]
    prey_stats = [PreyStat() for i in 1 : cf.num_preys]

    for step in 1 : num_steps
        ob_predators = [get_observation(s, p) for p in s.predators]
        a_predators = [res.predator(model_predator, ob, true) for ob in ob_predators]
        ob_preys = [get_observation(s, p) for p in s.preys]
        a_preys = [res.prey(model_prey, ob, false) for ob in ob_preys]
        
        s_next, out_predators, out_preys = forward(s, a_predators, a_preys)
        
        for (i, y) in enumerate(out_predators)
            push!(predator_stats[i].reward, Q(y, true))
            if y[1] == 1
                predator_stats[i].eat_count += 1
            end
        end
        
        for (i, y) in enumerate(out_preys)
            push!(prey_stats[i].reward, Q(y, false))
            N = length(prey_stats[i].life_time)
            prey_stats[i].life_time[N] += 1
            if y[1] == 1
                push!(prey_stats[i].life_time, 0)
            end
        end

        if step % 10000 == 0 log_simulate_step(log_prefix, name, step) end
        s = s_next
    end
    
    stats = (predator=predator_stats, prey=prey_stats)
    @save log_prefix * ".bson" stats
end

println("Simulate on config " * ARGS[1])
cf = get_config(ARGS[1])
log_path = "./log/" * ARGS[1] * "/"
model_path = log_path * "/model/"
num_steps = parse(Int64, ARGS[2])

# simulate(cf, (predator=best_response, prey=best_response), "BB")
# simulate(cf, (predator=best_response, prey=softmax_response), "BS")
# simulate(cf, (predator=best_response, prey=random_response), "BR")

# simulate(cf, (predator=softmax_response, prey=best_response), "SB")
simulate(cf, (predator=softmax_response, prey=softmax_response), "SS")
simulate(cf, (predator=softmax_response, prey=random_response), "SR")

# simulate(cf, (predator=random_response, prey=best_response), "RB")
simulate(cf, (predator=random_response, prey=softmax_response), "RS")
simulate(cf, (predator=random_response, prey=random_response), "RR")