include("config.jl")
using Flux
using CUDA

function get_nn(cf::Config)::Chain
    layers = []
    N = length(cf.hidden_layers)
    push!(layers, Dense(2*(cf.num_predators + cf.num_preys + 1), cf.hidden_layers[1], relu))
    for i in 1 : N - 1
        push!(layers, Dense(cf.hidden_layers[i], cf.hidden_layers[i+1], relu))
    end
    push!(layers, Dense(cf.hidden_layers[N], 1))
    return gpu(Chain(layers...))
end

function generate_input(ob::Array{Float64}, a::Tuple{Int64, Int64})::Array{Float64}
    x, y = get_vector(0, 0, a...)
    return [reshape(ob, :); x; y]
end

function Q(nn::Chain, ob::Array{Float64}, a::Tuple{Int64, Int64})
    x = generate_input(ob, a)
    x = gpu(x)
    return nn(x)
end

function choose_action(nn::Chain, ob::Array{Float64})::Tuple{Tuple{Int64, Int64}, Float64, Float64}
    a = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1), (0, -2), (-2, 0)]
    q_values::Array{Float64} = []
    for i in 1:7
        push!(q_values, Array(Q(nn, ob, a[i]))[1])
    end
    probs = softmax(q_values)
    idx = sample([1,2,3,4,5,6,7], Weights(probs))
    return a[idx], probs[idx], q_values[idx]
end