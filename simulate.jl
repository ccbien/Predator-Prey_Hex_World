include("config.jl")
include("markov_game.jl")
include("model.jl")
include("logger.jl")
using Flux
using CUDA
using BSON: @save, @load

mutable struct PredatorStat
    reward::Vector{Float64}
    eat_count::Int64
    PredatorStat() = new([], 0)
end

mutable struct PreyStat
    reward::Vector{Float64}
    PreyStat() = new([])
end

function simulate(cf::Config, res::NamedTuple, name::String)
    @load model_path*"predator.bson" model_predator
    @load model_path*"prey.bson" model_prey

    # Avoid CUDA bug
    try model_predator.nn(gpu(rand(2*(cf.num_predators + cf.num_preys + 1)))) catch end

    num_steps = parse(Int64, ARGS[2])
    s = get_random_state(cf)
    
    predator_stats = [PredatorStat() for i in 1 : cf.num_predators]
    prey_stats = [PreyStat() for i in 1 : cf.num_preys]
    predator_ids = Dict(i => i for i in 1:cf.num_predators)
    prey_ids = Dict(i => i for i in 1:cf.num_preys)

    for step in 1 : num_steps
        ob_predators = [get_observation(s, p) for p in s.predators]
        a_predators = [res.predator(model_predator, ob) for ob in ob_predators]
        ob_preys = [get_observation(s, p) for p in s.preys]
        a_preys = [res.prey(model_prey, ob) for ob in ob_preys]

        s_next, rw1, rw2, eat, is_alive = forward(s, a_predators, a_preys)

        for (i, j) in predator_ids
            push!(predator_stats[j].reward, rw1[i])
            if eat[i]
                predator_stats[j].eat_count += 1
            end
        end
            
        for (i, j) in prey_ids
            push!(prey_stats[j].reward, rw2[i])
            if !is_alive[i]
                push(prey_stats, PreyStat())
                prey_ids[i] = lenght(prey_stats)
            end
        end

        s = s_next
    end

    stats = (predator=predator_stats, prey=prey_stats)
    @save log_path * "sim_" * name * "_" * string(num_steps) * ".bson" stats
end

cf = get_config(ARGS[1])
log_path = "./log/" * ARGS[1] * "/"
model_path = log_path * "/model/"

simulate(cf, (predator=best_response, prey=best_response), "BB")
simulate(cf, (predator=best_response, prey=random_response), "BR")
simulate(cf, (predator=random_response, prey=best_response), "RB")