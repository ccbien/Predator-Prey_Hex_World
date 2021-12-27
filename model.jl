include("config.jl")
using Flux
using CUDA

struct QEstimator
    model
end

function get_QEstimator(cf::Config)
    layers = []
    N = length(cf.hidden_layers)
    push!(layers, Dense(2*(cf.num_predators + cf.num_preys), cf.hidden_layers[1], relu))
    for i in 1 : N - 1
        push!(layers, Dense(cf.hidden_layers[i], cf.hidden_layers[i+1], relu))
    end
    push!(layers, Dense(cf.hidden_layers[N], 7, relu))
    model = gpu(Chain(layers...))
    return QEstimator(model)
end

function get_Q_values(Q::QEstimator, o::Array{Float64})
    x = reshape(o, :)
    x = gpu(x)
    return Q.model(x)
end

function get_action_prob(Q::QEstimator, o::Array{Float64})
    q_values = get_Q_values(Q, o)
    return softmax(q_values)
end