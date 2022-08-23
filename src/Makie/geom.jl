function to_mesh(l::Lattice{2}, zs = nothing)
    return to_mesh(MonteCarlo.lattice_vectors(l), zs)
end

function to_reciprocal_mesh(l::Lattice{2}, zs = nothing)
    return to_mesh(MonteCarlo.reciprocal_vectors(l), zs)
end

function to_reciprocal_mesh(vs::Tuple{2, <: Vector}, zs::Matrix)
    v1, v2 = vs
    Lx, Ly = size(zs)
    if zs === nothing
        ps = Point2f[i * v1 + j * v2 for i in 0:Lx-1 for j in 0:Ly-1]
    else
        ps = [Makie.to_ndim(Point3f, (i-1) * v1 + (j-1) * v2, zs[i, j]) for i in 0:Lx for j in 0:Ly]
    end

    if dot(v1, v2) ≈ 0 # 90°
        fs = [
            QuadFace(i + Lx * (j-1), i+1 + Lx * (j-1), i+1 + Lx * j, i + Lx * j) 
            for i in 1:Lx-1 for j in 1:Ly-1
        ]
    elseif abs(dot(v1, v2)) ≈ norm(q1) * norm(q2) * 0.5 # 60°
        fs = vcat([
            GLTriangleFace(i + Lx * (j-1), i+1 + Lx * (j-1), i + Lx * j) 
            for i in 1:Lx-1 for j in 1:Ly-1
        ], [
            GLTriangleFace(i+1 + Lx * (j-1), i + 1 + Lx * j, i + Lx * j) 
            for i in 1:Lx-1 for j in 1:Ly-1
        ])
    else # Honeycomb lattices aren't real and they can't hurt you
        error("No face generation implemented for this lattice.")
    end

    return normal_mesh(ps, fs)
end

sq_dist_2d(p1) = p1[1]^2 + p1[2]^2

function to_wireframe(qs, vals)
    ps = Point3f.(first.(qs), last.(qs), vals)
    frame = Point3f[]
    mindist = minimum(sq_dist_2d, (ps[1],) .- ps[2:end])
    
    for i in eachindex(ps)
        for j in i+1:length(ps)
            if sq_dist_2d(ps[i] - ps[j]) < 1.1mindist
                push!(frame, ps[i], ps[j])
            end
        end
    end
    
    return frame
end