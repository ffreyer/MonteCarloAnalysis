
"""
    tile2D(dqmc)
    tile2D(corners)

Returns a 2D tile (mesh) that can seamlessly fill the reciprocal lattice of a 
given simulation. 

Note that the returned mesh needs to be scaled with `(1/L, 1/L)`. This is done
to allow/simplify using the same mesh for multiple system sizes.
"""
tile2D(dqmc::DQMC) = tile2D(get_sorted_corners(lattice(dqmc)))
function tile2D(cs::Vector)
    L = length(cs)
    vertices = [[Point3f0(0, 0, 1)]; map(v -> Point3f0(v..., 1), cs)]
    faces = [TriangleFace(1, ((i+1)%L)+2, (i%L)+2) for i in 1:L]
    GeometryBasics.normal_mesh(vertices, faces)
end


"""
    tile3D(dqmc)
    tile3D(corners)

Returns an extruded tile (mesh) that can seamlessly fill the reciprocal lattice
of a given simulation

Note that the returned mesh needs to be scaled with `(1/L, 1/L, height)`. This 
is done to allow/simplify using the same mesh for multiple system sizes.
"""
tile3D(dqmc::DQMC) = tile3D(MonteCarlo.get_sorted_corners(lattice(dqmc)))
function tile3D(cs::Vector)
    L = length(cs)
    vertices = [
        # bottom
        [Point3f0(0)]; map(v -> Point3f0(v..., 0), cs);
        # sides
        map(v -> Point3f0(v..., 0), cs); map(v -> Point3f0(v..., 0), cs); 
        map(v -> Point3f0(v..., 1), cs); map(v -> Point3f0(v..., 1), cs);
        # top
        [Point3f0(0, 0, 1)]; map(v -> Point3f0(v..., 1), cs) 
    ]

    faces = [
        [TriangleFace(1, ((i+1)%L)+2, (i%L)+2) for i in 1:L];
        [QuadFace(
            (i%L) +L +2, ((i+1)%L) +2L +2, ((i+1)%L) +3L +2, (i%L) +4L +2
        ) for i in 1:L];
        [TriangleFace(5L+2, (i%L) +5L +3, ((i+1)%L) +5L +3) for i in 1:L];
    ]

    GeometryBasics.normal_mesh(vertices, faces)
end


# TODO
# How do I wanna add tiled histograms?