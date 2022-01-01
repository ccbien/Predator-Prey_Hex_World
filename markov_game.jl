include("config.jl")
include("utils.jl")
include("model.jl")
using Flux: softmax
using StatsBase: sample, Weights

struct State
    n_rows::Integer
    n_cols::Integer
    predators::Vector{Tuple{Int64, Int64}}
    preys::Vector{Tuple{Int64, Int64}}
end

function get_all_actions()
    return [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, -2), (0, 2)]
end

function get_random_state(cf::Config)::State
    n_rows = cf.n_rows
    n_cols = cf.n_cols
    
    set = Set()
    predators = []
    while length(predators) < cf.num_predators
        r = rand(1:n_rows)
        c = rand(1:n_cols)
        if (r + c) % 2 == 1
            continue
        end
        push!(set, (r, c))
        push!(predators, (r, c))
    end

    preys = []
    while length(preys) < cf.num_preys
        r = rand(1:n_rows)
        c = rand(1:n_cols)
        if (r + c) % 2 == 1 || (r, c) in set
            continue
        end
        push!(preys, (r, c))
    end

    @assert(length(intersect(Set(predators), Set(preys))) == 0)
    return State(n_rows, n_cols, predators, preys)
end

function get_observation(state::State, anchor::Tuple{Int64, Int64})::Matrix{Float64}
    M = length(state.predators)
    N = length(state.preys)
    ob = Array{Float64, 2}(undef, M + N + 1, 2)
    r0, c0 = anchor

    temp = []
    idx = 0
    for (r, c) in state.predators
        idx += 1
        x, y = get_vector(r0, c0, r, c)
        ob[idx, :] = [x, y]
        push!(temp, x^2 + y^2)
    end
    for (r, c) in state.preys
        idx += 1
        x, y = get_vector(r0, c0, r, c)
        ob[idx, :] = [x, y]
        push!(temp, x^2 + y^2)
    end

    temp[1 : M] = sortperm(temp[1 : M])
    temp[M + 1 : M + N] = sortperm(temp[M + 1 : M + N]) .+ M
    ob[1:M+N, :] = ob[temp, :]
    ob[M+N+1, :] .= get_vector(0, 0, state.n_rows - 2*r0, state.n_cols - 2*c0)
    return ob
end

function softmax_response(model::Model, ob::Matrix{Float64}, is_predator::Bool)::Tuple{Int64, Int64}
    a = get_all_actions()
    q_values = [Q(model, ob, a[i], is_predator) for i in 1:7]
    probs = softmax(q_values)
    idx = sample(1:7, Weights(probs))
    return a[idx]
end

function best_response(model::Model, ob::Matrix{Float64}, is_predator::Bool)::Tuple{Int64, Int64}
    a = get_all_actions()
    q_values = [Q(model, ob, a[i], is_predator) for i in 1:7]
    idx = argmax(q_values)
    return a[idx]
end

function random_response(model::Model, ob::Matrix{Float64}, is_predator::Bool)
    return sample(get_all_actions())
end

function forward(s::State, a_predators::Vector{Tuple{Int64, Int64}}, a_preys::Vector{Tuple{Int64, Int64}})    
    rewards = get_reward_dict()
    out_predators = [[-1., 0.] for a in a_predators]
    out_preys = [[-1., 0.] for a in a_preys]

    s_next = deepcopy(s)
    for (i, (r, c)) in enumerate(s.predators)
        r, c = s.predators[i]
        dr, dc = a_predators[i]
        if is_outside(s, r + dr, c + dc) dr = dc = 0 end 
        s_next.predators[i] = (r + dr, c + dc)
    end
    for (i, (r, c)) in enumerate(s.preys)
        r, c = s.preys[i]
        dr, dc = a_preys[i]
        if is_outside(s, r + dr, c + dc) dr = dc = 0 end
        s_next.preys[i] = (r + dr, c + dc)
    end

    count_predators = get_count_dict(s_next.predators)
    count_preys = get_count_dict(s_next.preys)
    for (i, item) in enumerate(s_next.predators)
        out_predators[i][1] = (get(count_preys, item, 0) / get(count_predators, item, 0))*2 - 1
    end
    for (i, item) in enumerate(s_next.preys)
        if get(count_predators, item, 0) > 0
            out_preys[i][1] = 1.0
            while true
                r = rand(1:s.n_rows)
                c = rand(1:s.n_cols)
                if (r + c) % 2 == 1 || get(count_predators, (r, c), 0) > 0
                    continue
                end
                s_next.preys[i] = (r, c)
                break
            end
        end
    end

    is_alive = [x[1] == -1 for x in out_preys]
    for (i, ((r, c), (r_next, c_next))) in enumerate(zip(s.predators, s_next.predators))
        if out_predators[i][1] == 0
            min_steps = get_distance_to_the_nearest(r, c, s.preys[is_alive])
            min_steps_next = get_distance_to_the_nearest(r_next, c_next, s_next.preys[is_alive])
            if min_steps_next > min_steps
                out_predators[i][2] = 1
            elseif min_steps_next < min_steps
                out_predators[i][2] = -1
            else
                out_predators[i][2] = 0
            end
        end
    end
    for (i, ((r, c), (r_next, c_next))) in enumerate(zip(s.preys, s_next.preys))
        if out_preys[i][1] == 0
            min_steps = get_distance_to_the_nearest(r, c, s.predators)
            min_steps_next = get_distance_to_the_nearest(r_next, c_next, s_next.predators)
            if min_steps_next > min_steps
                out_preys[i][2] = 1
            elseif min_steps_next < min_steps
                out_preys[i][2] = -1
            else
                out_preys[i][2] = 0
            end
        end
    end

    return s_next, out_predators, out_preys
end

function get_utility(s::State, model::Model, agents::Vector{Tuple{Int64, Int64}}, is_predator::Bool)::Vector{Float64}
    obs = [get_observation(s, agent) for agent in agents]
    us = []
    for ob in obs
        q_values = [Q(model, ob, a, is_predator) for a in get_all_actions()]
        probs = softmax(q_values)
        u = 0
        for (a, p) in zip(get_all_actions(), probs)
            u += Q(model, ob, a, is_predator) * p
        end
        push!(us, u)
    end
    return us
end