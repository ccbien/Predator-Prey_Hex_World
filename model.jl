include("config.jl")
using Flux: Chain, Dense, gpu, ADAM
struct Model
    nn::Chain
    opt::ADAM
end

function get_model(cf::Config)::Model
    layers = []
    N = length(cf.hidden_layers)
    push!(layers, Dense(2*(cf.num_predators + cf.num_preys + 1), cf.hidden_layers[1], relu))
    for i in 1 : N - 1
        push!(layers, Dense(cf.hidden_layers[i], cf.hidden_layers[i+1], relu))
    end
    push!(layers, Dense(cf.hidden_layers[N], 1))
    nn = gpu(Chain(layers...))
    opt = gpu(ADAM(cf.η, cf.β))
    return Model(nn, opt)
end

function generate_input(ob::Matrix{Float64}, a::Tuple{Int64, Int64})::Vector{Float64}
    x, y = get_vector(0, 0, a...)
    return [reshape(ob, :); x; y]
end

function train_step(model::Model, ob::Matrix{Float64}, a::Tuple{Int64, Int64}, rw::Vector{Float64})::Float64
    x = generate_input(ob, a)

    # loss, gradient
end

function Q(model::Model, ob::Matrix{Float64}, a::Tuple{Int64, Int64})::Float64
    x = generate_input(ob, a)
    x = gpu(x)
    return Array(model.nn(x))[1]
end