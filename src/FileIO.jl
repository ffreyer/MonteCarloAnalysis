using MonteCarlo.BinningAnalysis

"""
    ValueWrapper(mc, ::Val{key})
    ValueWrapper(mc, m::AbstractMeasurement)

Attempt to calculate a (value, error) pair for a given measurement. You can call
`mean` and `std_error` to get those values.



The conversion is split into a few simple, generic methods so that the default behavior
can be adjusted at different points. The call stack from `load` is:

1. `simplify_measurements!(mc)` calls `simplify(ms, mc, ::Val{key})` with each
    measurement key to populate a new measurements Dict `ms` which will replace
    the old one.
2. `simplify(ms, mc, v::Val{key}) = ms[key] = ValueWrapper(mc, v)`
3. `ValueWrapper(mc, ::Val{key}) = ValueWrapper(mc, mc[key])`
4. `ValueWrapper(mc, measurement) = ValueWrapper(mean(measurement), std_error(measurement))`

Example Modifications:
- If you want to further process a specifically typed set of measurements you 
    can add a method for (4)
- Replacing the default behavior for a specific key can be done at (2) or (3).
- Adjusting the name of a measurement or splitting one can be done at (2)
- Removing measurements can be done at (2) by not inserting the value in `ms`.
"""
struct ValueWrapper{T1, T2} <: AbstractMeasurement
    exp_value::T1
    std_error::T2
end

function simplify_measurements!(mc)
    ms = Dict{Symbol, AbstractMeasurement}()
    for key in keys(mc)
        simplify(ms, mc, Val(key))
    end
    mc.measurements = ms
    nothing
end
simplify(ms, mc, v::Val{key}) where key = ms[key] = ValueWrapper(mc, v)
ValueWrapper(mc, ::Val{key}) where key = ValueWrapper(mc, mc[key])
function ValueWrapper(mc::MonteCarloFlavor, m::AbstractMeasurement)
    return ValueWrapper(mean(m), std_error(m))
end
BinningAnalysis.mean(v::ValueWrapper) = v.exp_value
BinningAnalysis.std_error(v::ValueWrapper) = v.std_error
Base.show(io::IO, ::MIME"text/plain", m::ValueWrapper) = show(io, m)
function Base.show(io::IO, m::ValueWrapper)
    print(io, m.exp_value)
    print(io, " ± ")
    print(io, m.std_error)
end


# get all files in directory recursively
function to_files(path_or_filename)
    if isfile(path_or_filename)
        return [path_or_filename]
    elseif isdir(path_or_filename)
        return vcat(to_files.(readdir(path_or_filename, join=true))...)
    else
        error("$path_or_filename is neither a valid directory nor file path.")
    end
end

function MonteCarlo.load(
        paths_or_filenames::Vector{String}; 
        prefix = "", postfix = r"jld2", simplify = false, silent = false,
        parallel = true, on_error = e -> @error(exception = e),
        lattice_instancing = true
    )
    # Normalize input to filepaths (recursively)
    files = String[]
    for path_or_file in paths_or_filenames
        _files = to_files(path_or_file)
        filter!(_files) do filepath
            _, filename = splitdir(filepath)
            startswith(filename, prefix) && endswith(filename, postfix)
        end
        append!(files, _files)
    end

    println(
        "Loading $(length(files)) Simulations", 
        parallel && nprocs() > 1 ? " on $(nworkers()) workers" : ""
    )
    flush(stdout)

    instances = MonteCarlo.AbstractLattice[]

    # Might be worth shuffling files for more equal load times?
    mcs = ProgressMeter.@showprogress pmap(files, distributed = parallel, on_error = on_error) do f
        mc = load(f)
        if lattice_instancing
            idx = findfirst(l -> lattice(mc) == l, instances)
            if idx === nothing
                push!(instances, lattice(mc))
            else
                mc.model = Model(mc.model, l = instances[idx::Int])
            end
        end
        simplify && simplify_measurements!(mc)
        mc
    end

    # fix potential dublications
    # we still want instancing in the parallel loop though, since simplify may 
    # benefit from it
    if lattice_instancing
        instances = MonteCarlo.AbstractLattice[]
        for mc in mcs
            idx = findfirst(l -> lattice(mc) == l, instances)
            if idx === nothing
                push!(instances, lattice(mc))
            else
                mc.model = Model(mc.model, l = instances[idx::Int])
            end
        end
    end

    return mcs
end
