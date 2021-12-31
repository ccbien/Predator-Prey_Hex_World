include("config.jl")
using Dates: now, format
using Statistics: mean
using Printf: @printf

mutable struct PredatorStat
    reward::Vector{Float64}
    eat_count::Int64
    PredatorStat() = new([], 0)
end

mutable struct PreyStat
    reward::Vector{Float64}
    PreyStat() = new([])
end

function log_config(log_path::String, cf::Config)
    open(log_path * "train_log.txt", "a") do out
        redirect_stdout(out) do
            println(cf)
        end
    end
end

function log_train_step(log_path, iter, step, losses, rewards)
    time = string(format(now(), "YYYY-mm-dd_HH:MM:SS"))
    predator_loss = losses.predator[(iter, step)]
    prey_loss = losses.prey[(iter, step)]
    predator_rw = rewards.predator[(iter, step)]
    prey_rw = rewards.prey[(iter, step)]
    vals = (time, iter, step, predator_loss, prey_loss, predator_rw, prey_rw)
    open(log_path * "train_log.txt", "a") do out
        redirect_stdout(out) do
            @printf("%s | iter=%6d | step=%6d | predator_loss=%11.4f | prey_loss=%11.4f | predator_rw=%8.2f | prey_rw=%8.2f |\n", vals...)
        end 
    end
    @printf("%s | iter=%6d | step=%6d | predator_loss=%11.4f | prey_loss=%11.4f | predator_rw=%8.2f | prey_rw=%8.2f |\n", vals...)
end

function log_train_iteration(log_path, iter, num_steps, losses, rewards)
    time = string(format(now(), "YYYY-mm-dd_HH:MM:SS"))
    predator_loss = mean([losses.predator[(iter, step)] for step in 1:num_steps])
    prey_loss = mean([losses.prey[(iter, step)] for step in 1:num_steps])
    predator_rw = mean([rewards.predator[(iter, step)] for step in 1:num_steps])
    prey_rw = mean([rewards.prey[(iter, step)] for step in 1:num_steps])
    vals = (time, iter, predator_loss, prey_loss, predator_rw, prey_rw)
    open(log_path * "train_log.txt", "a") do out
        redirect_stdout(out) do
            @printf("%s | iter=%6d | predator_loss=%11.4f | prey_loss=%11.4f | predator_rw=%8.2f | prey_rw=%8.2f |\n", vals...)
        end 
    end
    @printf("%s | iter=%6d | predator_loss=%11.4f | prey_loss=%11.4f | predator_rw=%8.2f | prey_rw=%8.2f |\n", vals...)
end

function log_train_info(log_path, line)
    time = string(format(now(), "YYYY-mm-dd_HH:MM:SS"))
    open(log_path * "train_log.txt", "a") do out
        redirect_stdout(out) do
            @printf("%s | %s\n", time, line)
        end 
    end
    @printf("%s | %s\n", time, line)
end

function log_simulate_step(log_prefix, name, step)
    time = string(format(now(), "YYYY-mm-dd_HH:MM:SS"))
    open(log_prefix * ".txt", "a") do out
        redirect_stdout(out) do
            @printf("%s | Done step #%d\n", time, step)
        end 
    end
    @printf("%s | %s | Done step #%d\n", time, name, step)
end