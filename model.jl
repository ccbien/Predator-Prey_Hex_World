include("config.jl")
using Flux: Chain, Dense, relu, σ, ADAM, params, gradient, mse, mae, update!
struct Model
    nn::Chain
    opt::ADAM
end

function get_model(cf::Config)::Model
    layers = []
    N = length(cf.hidden_layers)
    push!(layers, Dense(2*(cf.num_predators + cf.num_preys + 2), cf.hidden_layers[1], σ))
    for i in 1 : N - 1
        push!(layers, Dense(cf.hidden_layers[i], cf.hidden_layers[i+1], σ))
    end
    push!(layers, Dense(cf.hidden_layers[N], 1))
    nn = Chain(layers...)
    opt = ADAM(cf.η, cf.β)
    return Model(nn, opt)
end

function generate_input(ob::Matrix{Float64}, a::Tuple{Int64, Int64})::Vector{Float64}
    ob = deepcopy(ob)
    for i in 1 : size(ob)[1]
        ob[i, :] .= normalize_vector(ob[i, :]...)
    end
    x, y = normalize_vector(get_vector(0, 0, a...)...)
    return [reshape(ob, :); x; y]
end

function train_step(model::Model, ob::Matrix{Float64}, a::Tuple{Int64, Int64}, y_true::Float64)::Float64
    x = generate_input(ob, a)
    y_true = [y_true]
    parameters = params(model.nn)

    function mse_loss(x, y_true)
        y_pred = model.nn(x)
        return mse(y_pred, y_true)
    end

    function mae_loss(x, y_true)
        y_pred = model.nn(x)
        return mae(y_pred, y_true)
    end
    
    loss = 0
    grads = gradient(parameters) do
        loss = mae_loss(x, y_true)
    end
    update!(model.opt, parameters, grads)
    return loss
end

function Q(model::Model, ob::Matrix{Float64}, a::Tuple{Int64, Int64})::Float64
    x = generate_input(ob, a)
    return Array(model.nn(x))[1]
end