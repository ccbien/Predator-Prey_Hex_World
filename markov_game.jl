include("config.jl")
include("utils.jl")
include("model.jl")
using Flux: softmax
using StatsBase: sample, Weights

struct State
    n_rows::Integer
    n_cols::Integer
    predators::Array{Tuple{Int64, Int64}}
    preys::Array{Tuple{Int64, Int64}}
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

function get_observation(state::State, anchor::Tuple{Int64, Int64})::Array{Float64}
    M = length(state.predators)
    N = length(state.preys)
    ob = Array{Float64, 2}(undef, M + N, 2)
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

    ob = ob[temp, :]
    return ob
end

function choose_action(nn::Chain, ob::Array{Float64})::Tuple{Tuple{Int64, Int64}, Float64, Float64}
    a = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, -2), (-2, 0)]
    q_values = []
    for i in 1:7
        push!(q_values, Q(nn, ob, a[i]))
    end
    probs = softmax(q_values)
    idx = sample([1,2,3,4,5,6,7], Weights(probs))
    return a[idx], probs[idx], q_values[idx]
end