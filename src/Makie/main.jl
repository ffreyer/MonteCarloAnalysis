using Makie, Colors, Observables, LsqFit, Loess, GeometryBasics, Printf
import Makie: convert_arguments, PointBased
using MonteCarlo: Bond, OpenBondIterator, bonds_open

include("colors.jl")

# Takes a DataFrame and builds some sort of interactive window to view all the 
# data
include("dataset.jl")

# 3D histograms (or do they call them 2d?) for CDC etc (TODO)
include("tiled_hist2d.jl")

# lattice to wireframe and mesh and stuff
include("geom.jl")

# linesegments(lattice(mc)) and whatnot
include("extension.jl")