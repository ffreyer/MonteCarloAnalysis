function convert_arguments(P::Type{<:Scatter}, l::Lattice{D}) where D
    return convert_arguments(P, Point{D}.(positions(l))[:])
end
function convert_arguments(P::Type{<:LineSegments}, l::Lattice{D}) where D
    return convert_arguments(P, collect(bonds_open(l)), Point{D}.(positions(l))[:])
end

function convert_arguments(
        P::Type{<:LineSegments}, 
        bonds::Union{OpenBondIterator, Array}, 
        sites::Union{Base.Generator, Array}
    )
    convert_arguments(P, collect(bonds)[:], collect(sites)[:])
end

function convert_arguments(
        P::Type{<:LineSegments}, bonds::Vector{<: Bond}, sites::Vector
    )
    T = Point{length(sites[1]), Float32}
    ps = [T(sites[i]) for b in bonds for i in [b.from, b.to]]
    convert_arguments(P, ps)
end