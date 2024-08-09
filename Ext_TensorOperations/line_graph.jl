using TensorOperations, OMEinsumContractionOrders, KaHyPar
using Graphs
using BenchmarkTools
using TensorOperations: optimaltree, TreeOptimizer, Poly
using CSV, DataFrames
using Plots

function line_network(n)
    network = [[i, (i + 1)] for i in 1:n - 1]
    return network
end
eval_ploy(p::Poly, x) = sum([p.coeffs[i] * x^(i-1) for i in 1:length(p.coeffs)])

time_tws = []
time_ncons = []

for n in 2:30
    cost = TensorOperations.Power{:χ}(1, 1)
    optdata_ncon = Dict([i => cost for i in 1:n])
    time_ncon = @belapsed (optimaltree($(line_network(n)), $optdata_ncon, TreeOptimizer{:NCon}(), false))

    optdata = Dict([i => 2 for i in 1:n])
    time_tw = @belapsed (optimaltree($(line_network(n)), $optdata, TreeOptimizer{:ExactTreewidth}(), false))

    @show n, time_ncon, time_tw
    push!(time_tws, time_tw)
    push!(time_ncons, time_ncon)
end

ns = [2:30...]
fig = plot(log10.(ns), log10.(time_tws), label="ExactTreewidth", xlabel="log(n)", ylabel="log(t)", title="Time vs n")
plot!(log10.(ns), log10.(time_ncons), label="NCon")
savefig(fig, "line_graph.png")