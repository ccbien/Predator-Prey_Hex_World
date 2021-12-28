using Dates: now, format
using Statistics: mean
using Printf: @printf

function log_config(log_path::String, cf::Config)
    open(log_path * "train_log.txt", "a") do out
        redirect_stdout(out) do
            println(cf)
        end
    end
end

function log_train_step(log_path, iter, step, losses, rewards)
    open(log_path * "train_log.txt", "a") do out
        redirect_stdout(out) do
            time = string(format(now(), "YYYY-mm-dd_HH:MM:SS"))
            predator_loss = losses.predator[(iter, step)]
            prey_loss = losses.prey[(iter, step)]
            predator_rw = rewards.predator[(iter, step)]
            prey_rw = rewards.prey[(iter, step)]
            vals = (time, iter, step, predator_loss, prey_loss, predator_rw, prey_rw)
            @printf("%s | iter=%6d | step=%6d | predator_loss=%11.4f | prey_loss=%11.4f | predator_rw=%8.2f | prey_rw=%8.2f |", vals...)
            println()
        end 
    end
end

function log_info(log_path, line)
    open(log_path * "train_log.txt", "a") do out
        redirect_stdout(out) do
            time = string(format(now(), "YYYY-mm-dd_HH:MM:SS"))
            @printf("%s | %s", time, line)
            println()
        end 
    end
end