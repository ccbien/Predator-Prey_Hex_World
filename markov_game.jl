include("config.jl")
include("utils.jl")
include("model.jl")

struct State
    n_rows::Integer
    n_cols::Integer
    predators::Array{Tuple{Int32, Int32}}
    preys::Array{Tuple{Int32, Int32}}
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

function get_observation(state::State, anchor::Tuple{Int32, Int32})::Array{Float64}
    M = length(state.predators)
    N = length(state.preys)
    ob = Array{Float64, 2}(undef, M + N, 2)
    r0, c0 = anchor
    x0, y0 = get_coordinate(r0, c0)

    temp = []
    idx = 0
    for (r, c) in state.predators
        idx += 1
        x, y = get_coordinate(r, c)
        ob[idx, :] = [x - x0, y - y0]
        push!(temp, (x - x0)^2 + (y - y0)^2)
    end
    for (r, c) in state.preys
        idx += 1
        x, y = get_coordinate(r, c)
        ob[idx, :] = [x - x0, y - y0]
        push!(temp, (x - x0)^2 + (y - y0)^2)
    end

    temp[1 : M] = sortperm(temp[1 : M])
    temp[M + 1 : M + N] = sortperm(temp[M + 1 : M + N]) .+ M

    ob = ob[temp, :]
    return ob
end

function get_action(nn::Chain, ob::Array{Float64})::Tuple{Int32, Int32}
    probs = π(nn, ob)
    d = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, -2), (-2, 0)]
    return d[argmax(probs)]
end

function get_action(qvals::Array{Float64})::Tuple{Int32, Int32}
    probs = π(qvals)
    d = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, -2), (-2, 0)]
    return d[argmax(probs)]
end

function forward(s::State)
    # ob_predators = get_observation.(s, s.predators)
    # ob_preys = get_observation.(s, s.preys)

    # act_predators = get_action.(model_predator, ob_predators)
    # act_preys = get_action.(model_prey, ob_preys)
    # # return new state and actions
end