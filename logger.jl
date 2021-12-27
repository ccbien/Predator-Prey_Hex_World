using Dates: now

function log_config(log_path::String, cf::Config)
    open(log_path * "info.txt", "w") do io
        write(io, string(cf))
    end
end

function log_line()
end

function log_train()
end