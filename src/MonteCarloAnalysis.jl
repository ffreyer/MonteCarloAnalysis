module MonteCarloAnalysis

using Reexport
import ProgressMeter

# Might as well make these available without an explicit outside `using`
@reexport using MonteCarlo
@reexport using DataFrames
@reexport using LinearAlgebra
@reexport using FFTW
using Distributed
using MonteCarlo.BinningAnalysis

using MonteCarlo:
    lattice_vectors, reciprocal_vectors, generate_combinations, _apply_wrap!,
    directed_norm2, init_hopping_matrices, hopping_directions, nflavors,
    DQMCMeasurement

include("FileIO.jl")
include("DataFrames.jl")
include("fourier.jl")
include("superfluid_stiffness.jl")

using Requires

function __init__()
    ProgressMeter.ijulia_behavior(:append)
    @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" begin
        include("Makie/main.jl")
    end
end

export cached_para_ccc, dia_K
export fourier

end
