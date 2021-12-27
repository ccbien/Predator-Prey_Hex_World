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

function generate_input(ob::Matrix{Float64}, a::Tuple{Int64, Int64})::Vector{Float64}
    x, y = get_vector(0, 0, a...)
    return [reshape(ob, :); x; y]
end

function Q(nn::Chain, ob::Matrix{Float64}, a::Tuple{Int64, Int64})::Float64
    x = generate_input(ob, a)
    x = gpu(x)
    return Array(nn(x))[1]
end