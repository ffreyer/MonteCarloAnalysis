function superfluid_stiffness(
        mc::DQMC, G::DQMCMeasurement, ccs::DQMCMeasurement; 
        shift_dir = [1., 0.]
    )

    normalize!(shift_dir)
    
    # Find index corresponding to the smallest jump in q-space parallel to shift_dir
    idx = let
        qs = cached_reciprocal_discretization(l)::Matrix{Vector{Float64}}
        ortho = [-shift_dir[2], shift_dir[1]]
        idx2dist = map(i -> i => dot(qs[i], ortho), eachindex(qs))
        filter!(t -> abs(t[2]) < 1e-6, idx2dist)
        minimum(last, idx2dist)[1]
    end

    Kx  = dia_K_x(mc, G, shift_dir)
    Λxx = para_ccc(mc, ccs, shift_dir)

    return 0.25 * (-Kx - Λxx[idx])
end

"""
    dia_K_x(mc, key::Symbol, dir::Vector)
    dia_K_x(mc, m::DQMCMeasurement, dir::Vector)
    dia_K_x(mc, G::Matrix, dir::Vector)

Computes the diamangetic contribution of electromagnetic response of the system
along a given direction `dir`. 
"""
dia_K_x(mc::DQMC, key::Symbol, shift_dir) = dia_K_x(mc, mean(mc[key]), shift_dir)
dia_K_x(mc::DQMC, m::DQMCMeasurement, shift_dir) = dia_K_x(mc, mean(m), shift_dir)
function dia_K_x(mc::DQMC, G::AbstractMatrix, shift_dir)
    # I'm heavily referring to and testing against
    # 1. https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.68.2830
    # 2. https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.201112
    #    (https://arxiv.org/pdf/1912.08848.pdf)

    # Get hoppings
    if !isdefined(mc.stack, :hopping_matrix)
        init_hopping_matrices(mc, mc.model)
    end
    T = Matrix(mc.stack.hopping_matrix)

    # Filter directions appropriate to shift_dir
    normalize!(shift_dir) # to be sure
    dirs = directions(lattice(mc))
    idxs = filter(hopping_directions(lattice(mc))) do i
        dot(dirs[i], shift_dir) > 1e-6
    end

    # Lattice iterator to map directional indices to (src, trg) pairs
    dir2srctrg = lattice(mc)[:dir2srctrg]
    N = length(lattice(mc))
    
    # If we used symmetry between spin up and spin down to reduce the size of 
    # the greens matrix we need to include a factor 2 to include both spins.
    flv = nflavors(mc)
    if     flv == 1; f = 2.0
    elseif flv == 2; f = 1.0
    else error("The diamagnetic contribution to the superfluid density has no implementation for $flv flavors")
    end
    
    Kx = 0.0

    # sum over spins
    for shift in 0 : N : flv*N - 1
        # sum over valid hopping directions
        for i in idxs
            # weight factor from distance
            # This comes from a derivative in the Taylor expansion of the Peirls
            # phase factor, see (2)
            weight = f * dot(dirs[i], shift_dir)^2

            for (src, trg) in dir2srctrg[i]
                # c_j^† c_i = δ_ij - G[i, j], but δ is always 0 because this 
                # excludes on-site. See (2)
                # Reverse directions are filtered out before this and explicitly
                # included again here.
                t = T[trg+shift, src+shift]
                Kx -= weight * (
                         t  * G[src+shift, trg+shift] + 
                    conj(t) * G[trg+shift, src+shift]
                )
            end
        end
    end

    # While (2) suggests L⁻² here it seems like they are actually using 1/N
    # based on comparing data.
    # (1) does not explicitly define the normalization for Kx, but does use 
    # 1/N for Λxx (without defining N)
    # Some other sources obfuscate the normalization as ⟨…⟩ or not include 
    # any normalization...
    return Kx / N
end


# This maybe usefull to deal with different `iter.directions`
_mapping(dirs::Union{Vector, Tuple}) = copy(dirs)

"""
    cached_para_ccc(mc, key::Symbol, dir)
    cached_para_ccc(mc, m::DQMCMeasurement, dir)
    cached_para_ccc(lattice, iter::AbstractLatticeIterator, ccs, dir)

Returns the Fourier transformed current current correlation along a given 
direction. This is the paramagnetic contribution of the electromagnetic response.
"""
function cached_para_ccc(mc::DQMC, key::Symbol, shift_dir; kwargs...)
    return cached_para_ccc(mc, mc[key], shift_dir; kwargs...)
end
function cached_para_ccc(mc::DQMC, m::DQMCMeasurement, shift_dir; kwargs...)
    return cached_para_ccc(lattice(mc), m.lattice_iterator, mean(m), shift_dir; kwargs...)
end

function cached_para_ccc(l::Lattice, iter::EachLocalQuadByDistance, ccs::Array, shift_dir)
    # Following Hofmann, Berg, Chowdhurry 
    # https://journals.aps.org/prb/abstract/10.1103/PhysRevB.102.201112 or
    # https://arxiv.org/pdf/1912.08848.pdf
    # - (1) relevant bonds have a positive component in `shift_dir` direction
    # - (2) have a prefactor dot(bond_dir, shift_dir)
    # - include there hermitian conjugate (done in simulation)
    # - are integrated over imaginary time at ω = 0 (done in simulation)
    # - (3) are Fourier transformed according to the Bravais lattice
    # - (4) get a Fourier weight based on the center of each direction
    # also tested against https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.68.2830
    
    # Create caches
    qs = cached_reciprocal_discretization(l)::Matrix{Vector{Float64}}
    dir2idxs = get!(l, :Bravais_dir2indices, Bravais_dir2indices)::Vector{Tuple{Int, Int}}
    fft_cache = if haskey(l.cache.cache, :fft_cache)
        l.cache.cache[:fft_cache]
    else
        l.cache.cache[:fft_cache] = Matrix{ComplexF64}(undef, l.Ls)
    end::Matrix{ComplexF64}
    dirs = cached_directions(l)::Vector{Vector{Float64}}

    # Filter directions appropriate to shift_dir (1)
    normalize!(shift_dir) # to be sure
    idx2dir = _mapping(iter.directions)
    filter!(p -> dot(dirs[p[2]], shift_dir) > 1e-6, idx2dir)

    # Prepare output and buffers
    # Note that references will generally talk about Λₓₓ which corresponds to 
    # shift_dir = [1, 0]. Since this code doesn't specifically use that direction
    # I'm calling with just Λ.
    Λ = zeros(ComplexF64, size(fft_cache))
    temp = zeros(length(l.Ls))
    temp2 = zeros(length(l.Ls))

    # ccs indexing: [basis1, basis2, Bravais dir, sub_dir1, sub_dir2]

    # iterate bond directions (index into ccs, index into directions)
    for (n, ndir) in idx2dir
        for (m, mdir) in idx2dir
            # bond centers (4)
            temp .= 0.5 .* (dirs[mdir] .- dirs[ndir])

            # prefactor from bond distance (2)
            weight = dot(dirs[mdir], shift_dir) * dot(dirs[ndir], shift_dir)

            # iterate basis site indices (of bond source sites)
            for j in axes(ccs, 2), i in axes(ccs, 1)

                fft_cache .= 0.0

                # (index into directions, (i_1, i_2)) where
                # R = R_0 + i_1 * a_1 + i_2 * a_2
                # with R_0 the basis position, a lattice vectors
                for (dir, (x, y)) in enumerate(dir2idxs)
                    fft_cache[x, y] += ccs[i, j, dir, m, n]
                end

                # Fourier w.r.t Bravais lattice (3)
                fft!(fft_cache)

                # directions -> center
                # temp = 0.5(basis(trg) - basis(src) + dot(uc_shift, lattice_vectors))
                # center = 0.5(basis(trg) + basis(src) + dot(uc_shift, lattice_vectors))
                # so we need to shift by ± basis(src)
                temp2 .= temp .- l.unitcell.sites[j] .+ l.unitcell.sites[i]

                # Apply weights from directions (2, 4)
                for (idx, q) in enumerate(qs)
                    Λ[idx] += fft_cache[idx] * weight * cis(-dot(temp2, q))
                end
            end
        end
    end

    # The 1/N (1/L²?) factor is already taken care of during the simulation. 
    # If this factor is wrong one can easily adjust it after this.
    return Λ
end

function reverse_bond_table(l)
    # This generates a list reverse_bond[bond_idx]
    bs = l.unitcell.bonds
    table = zeros(Int, length(bs))

    for i in eachindex(table)
        if table[i] == 0
            idx = findfirst(bs) do b
                b.from == bs[i].to &&
                b.to   == bs[i].from &&
                b.uc_shift == .- bs[i].uc_shift
            end::Int

            table[i] = idx
            table[idx] = i
        end
    end

    return table
end

function cached_para_ccc(l::Lattice, iter::EachBondPairByBravaisDistance, ccs::Array, shift_dir; skip_check = false)
    # Equivalent to the above with a more straight forward lattice iterator
    
    # Get caches
    equivalency = get!(l, :reverse_bond_table, reverse_bond_table)
    fft_cache = if haskey(l.cache.cache, :fft_cache)
        l.cache.cache[:fft_cache]
    else
        l.cache.cache[:fft_cache] = Matrix{ComplexF64}(undef, l.Ls)
    end::Matrix{ComplexF64}
    qs = cached_reciprocal_discretization(l)::Matrix{Vector{Float64}}

    uc = l.unitcell
    bs = uc.bonds
    v1, v2 = lattice_vectors(l)

    # directions of all bonds
    ds = map(bs) do b
        uc.sites[b.to] - uc.sites[b.from] + v1 * b.uc_shift[1] + v2 * b.uc_shift[2]
    end

    # Bond centers relative to the Bravais lattice position
    cs = map(bs) do b
        0.5 * (uc.sites[b.to] + uc.sites[b.from] + v1 * b.uc_shift[1] + v2 * b.uc_shift[2])
    end

    # directions applicable to shift_dir (and measured bonds)
    applicable = let
        applicable = filter(i -> dot(ds[i], shift_dir) > 0, eachindex(ds))
        if iter.bond_idxs == Colon
            # All bonds are included, so the mapping is just i -> i
            Pair.(applicable, applicable)
        else
            # A subset of bonds is included. We need to further reduce this 
            # subset of bonds to those applicable to shift_dir. If bonds going 
            # in the reverse direction can act as replacements.
            output = Pair{Int, Int}[]

            for idx in applicable
                i = findfirst(isequal(idx), iter.bond_idxs)
                if i === nothing
                    i = findfirst(isequal(equivalency[idx]), iter.bond_idxs)
                    if i === nothing
                        if !skip_check
                            error("Missing bond $idx (or $(equivalency[idx]))")
                        else
                            continue
                        end
                    end
                    push!(output, i => equivalency[idx])
                else
                    push!(output, i => idx)
                end
            end

            output
        end
    end
    
    # Output
    Λ = zeros(ComplexF64, size(ccs, 1), size(ccs, 2))

    for (i, bi) in applicable, (j, bj) in applicable
        # Fourier transform on Bravais lattice for on pair of bonds (i, j)
        fft_cache[:, :] .= ccs[:, :, i, j]
        fft!(fft_cache)
        
        # weights from bond centers and directions
        weight = dot(ds[bi], shift_dir) * dot(ds[bj], shift_dir)
        for (idx, q) in enumerate(qs)
            Λ[idx] += fft_cache[idx] * weight * cis(-dot(cs[bi] - cs[bj], q))
        end
    end

    return Λ
end
