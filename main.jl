import Dates
include("config.jl")

cf = get_config_1()

log_path = "./log/" * string(Dates.now()) * "/"
mkpath(log_path)
open(log_path * "config.txt", "w") do io
    write(io, string(cf))
end

