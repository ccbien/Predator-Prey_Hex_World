include("config.jl")
using Flux
using CUDA

function get_nn(cf::Config)::Chain
    layers = []
    N = length(cf.hidden_layers)
    push!(layers, Dense(2*(cf.num_predators + cf.num_preys), cf.hidden_layers[1], relu))
    for i in 1 : N - 1
        push!(layers, Dense(cf.hidden_layers[i], cf.hidden_layers[i+1], relu))
    end
    push!(layers, Dense(cf.hidden_layers[N], 7, relu))
    return gpu(Chain(layers...))
end

function Q(nn::Chain, o::Array{Float64})
    x = reshape(o, :)
    x = gpu(x)
    return nn(x)
end

function π(nn::Chain, o::Array{Float64})
    qvals = get_qvals(nn, o)
    return softmax(q_values)
end

function π(qvals::Array{Float64})
    return softmax(qvals)
end
