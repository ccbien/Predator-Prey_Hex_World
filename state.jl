include("config.jl")
include("utils.jl")

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

function transition(s, a)
end

function get_observation(state::State, anchor)::Array{Float64}
    M = length(state.predators)
    N = length(state.preys)
    a = Array{Float64, 2}(undef, M + N, 2)
    r0, c0 = anchor
    x0, y0 = get_coordinate(r0, c0)

    idx = 0
    for (r, c) in state.predators
        idx += 1
        x, y = get_coordinate(r, c)
        a[idx, :] = [x - x0, y - y0]
    end
    for (r, c) in state.preys
        idx += 1
        x, y = get_coordinate(r, c)
        a[idx, :] = [x - x0, y - y0]
    end
    
    return a
end