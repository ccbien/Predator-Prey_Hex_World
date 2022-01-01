include("config.jl")
using Flux: Chain, Dense, relu, σ, tanh, ADAM, params, gradient, mse, mae, update!
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
    push!(layers, Dense(cf.hidden_layers[N], 2, tanh))
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

function train_step(model::Model, ob::Matrix{Float64}, a::Tuple{Int64, Int64}, y_true::Vector{Float64})::Float64
    function mse_loss(x, y_true)
        y_pred = model.nn(x)
        return mse(y_pred, y_true)
    end
    function mae_loss(x, y_true)
        y_pred = model.nn(x)
        return mae(y_pred, y_true)
    end

    x = generate_input(ob, a)
    parameters = params(model.nn)
    
    loss = 0
    grads = gradient(parameters) do
        loss = mse_loss(x, y_true)
    end
    update!(model.opt, parameters, grads)
    return loss
end

function get_reward_dict()
    return Dict( "predator_eat" => 20, "prey_eaten" => -100, "dist_factor" => 10)
end

function Q(model::Model, ob::Matrix{Float64}, a::Tuple{Int64, Int64}, is_predator::Bool)::Float64
    rw = get_reward_dict()
    x = generate_input(ob, a)
    y = model.nn(x)
    if is_predator
        return (y[1] + 1)/2 * rw["predator_eat"] - y[2] * rw["dist_factor"]
    else
        return (y[1] + 1)/2 * rw["prey_eaten"] + y[2] * rw["dist_factor"]
    end
end

function Q(y::Vector{Float64}, is_predator::Bool)::Float64
    rw = get_reward_dict()
    if is_predator
        return (y[1] + 1)/2 * rw["predator_eat"] - y[2] * rw["dist_factor"]
    else
        return (y[1] + 1)/2 * rw["prey_eaten"] + y[2] * rw["dist_factor"]
    end
end