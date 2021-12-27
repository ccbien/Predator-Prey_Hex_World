include("config.jl")
include("state.jl")
import Dates

function train(cf::Config)
    for iter in 1:cf.num_iterations
        s::State = get_random_state(cf)
        o = get_observation(s, s.predators[1])
        return
    end
end

cf = get_config_1()
log_path = "./log/" * string(Dates.now()) * "/"
mkpath(log_path)
open(log_path * "config.txt", "w") do io
    write(io, string(cf))
end

train(cf)