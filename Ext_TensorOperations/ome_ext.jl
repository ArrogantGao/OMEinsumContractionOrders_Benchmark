using TensorOperations, OMEinsumContractionOrders, KaHyPar
using Graphs
using BenchmarkTools
using TensorOperations: optimaltree, TreeOptimizer, Poly
using CSV, DataFrames

function graph2network(g::SimpleGraph)
    network = [Int[] for _=1:nv(g)]
    for (i, e) in enumerate(edges(g))
        u, v = src(e), dst(e)
        push!(network[u], i)
        push!(network[v], i)
    end
    return network
end

eval_ploy(p::Poly, x) = sum([p.coeffs[i] * x^(i-1) for i in 1:length(p.coeffs)])

CSV.write("ome_ext_small_graph.csv", DataFrame(graphname=[], nv=[], ne=[], NCon = [], ExactTreeWidth=[], GreedyMethod=[], KaHyParBipartite=[], SABipartite=[], TreeSA=[]))

for graphname in [:bull, :chvatal, :cubical, :diamond, :frucht, :heawood, :house, :housex, :krackhardtkite, :octahedral, :petersen, :sedgewickmaze, :tetrahedral, :truncatedtetrahedron]
    g = smallgraph(graphname)
    @show nv(g), ne(g), graphname
    network = graph2network(g)
    d = 2^4
    optdata = Dict([i => d for i in 1:ne(g)])
    
    time_complexity = []

    cost = TensorOperations.Power{:χ}(1, 1)
    optdata_ncon = Dict([i => cost for i in 1:ne(g)])
    tree, tc = optimaltree(network, optdata_ncon, TreeOptimizer{:NCon}(), true)
    tc_ncon = log2(eval_ploy(tc, d))

    for treeopt in [TreeOptimizer{:ExactTreewidth}(), TreeOptimizer{:GreedyMethod}(), TreeOptimizer{:KaHyParBipartite}(), TreeOptimizer{:SABipartite}(), TreeOptimizer{:TreeSA}()]
        @show treeopt
        tree, tc = optimaltree(network, optdata, treeopt, true)
        push!(time_complexity, log2(tc))
    end
    CSV.write("ome_ext_small_graph.csv", DataFrame(graphname=[graphname], nv=[nv(g)], ne=[ne(g)], NCon = tc_ncon, ExactTreeWidth=[time_complexity[1]], GreedyMethod=[time_complexity[2]], KaHyParBipartite=[time_complexity[3]], SABipartite=[time_complexity[4]], TreeSA=[time_complexity[5]]), append=true)
end

# on large graphs
CSV.write("C60.csv", DataFrame(method = [], tc = [], time = []))

C60_network = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [7, 16, 17], [18, 19, 20], [10, 21, 22], [23, 24, 25], [1, 16, 26], [27, 28, 29], [4, 21, 30], [13, 31, 32], [18, 31, 33], [26, 34, 35], [30, 36, 37], [38, 39, 40], [38, 41, 42], [39, 43, 44], [41, 45, 46], [47, 48, 49], [50, 51, 52], [23, 43, 53], [27, 45, 53], [32, 54, 55], [33, 56, 57], [34, 58, 59], [36, 60, 61], [62, 63, 64], [65, 66, 67], [17, 47, 68], [22, 50, 69], [48, 62, 70], [51, 63, 71], [54, 72, 73], [56, 72, 74], [73, 75, 76], [74, 77, 78], [75, 79, 80], [77, 79, 81], [2, 44, 82], [5, 46, 83], [8, 55, 84], [11, 57, 85], [70, 86, 87], [71, 86, 88], [68, 82, 89], [69, 83, 90], [35, 76, 84], [37, 78, 85], [58, 65, 80], [60, 66, 81], [24, 28, 67], [40, 87, 89], [42, 88, 90], [14, 19, 64], [9, 15, 49], [12, 20, 52], [3, 25, 59], [6, 29, 61]]
optdata = Dict([i => 2 for i in 1:90])

for treeopt in [TreeOptimizer{:GreedyMethod}(), TreeOptimizer{:KaHyParBipartite}(), TreeOptimizer{:SABipartite}(), TreeOptimizer{:TreeSA}()]
    tree, tc = optimaltree(C60_network, optdata, treeopt, false)
    time = @belapsed optimaltree($C60_network, $optdata, $treeopt, false)
    @show treeopt, log2(tc), time
    df = DataFrame(method=[string(treeopt)], tc=[log2(tc)], time=[time])
    CSV.write("C60.csv", df, append=true)
end

cost = TensorOperations.Power{:χ}(1, 1)
optdata_ncon = Dict([i => cost for i in 1:90])
time_ncon = @elapsed (result = optimaltree(C60_network, optdata_ncon, TreeOptimizer{:NCon}(), true))
tc_ncon = log2(eval_ploy(result[2], 2))
df = DataFrame(method=["NCon"], tc=[log2(tc_ncon)], time=[time_ncon])
CSV.write("C60.csv", df, append=true)