using Printf, LsqFit

function intmap(df, key)
    vals = unique(getproperty(df, key))
    Int[findfirst(isequal(x), vals)::Int for x in getproperty(df, key)]
end

function find_groups(df, _keys)
    sets = Dict{Symbol, Vector{Symbol}}()
    cache = Dict{Symbol, Vector{Int}}()
    
    for key in _keys
        idxs = intmap(df, key)
        for (k, v) in cache
            if v == idxs
                push!(sets[k], key)
                @goto next_iteration
            end
        end
        push!(sets, key => Symbol[])
        push!(cache, key => idxs)
        @label next_iteration
    end
    
    groups = [tuple(k, v...) for (k, v) in sets]
    
    return groups
end

# TODO someone else already did string formatting, check makie
function param_string(value::Int64, key)
    if value > 100_000_000
        @sprintf("%s = %iM", string(key), div(value, 1_000_000))
    elseif value > 10_000_000
        @sprintf("%s = %0.1fM", string(key), div(value, 1_000_000))        
    elseif value > 1_000_000
        @sprintf("%s = %0.2fM", string(key), div(value, 1_000_000))
    elseif value > 100_000
        @sprintf("%s = %ik", string(key), div(value, 1_000))
    elseif value > 10_000
        @sprintf("%s = %0.1fk", string(key), div(value, 1_000))        
    elseif value > 1_000
        @sprintf("%s = %0.2fk", string(key), div(value, 1_000))
    else
        @sprintf("%s = %i", string(key), div(value, 1_000))
    end
end
param_string(value::Float64, key) = @sprintf("%s = %0.2f", string(key), value)
param_string(value, key) = "$key = $value"
param_string(p::Pair) = param_string(p[2], p[1])
    
"""
    plot_GUI(df; kwargs...)

### Keyword Arguments
- `paramters` parameters in the dataframe to consider for groups
- `groups` list of (parameter, values)
- `group_sizes` number of elements in each group
- `max_menu_size`
- `represent_as = Dict(:beta => :ignore)` (:ignore, :menu, :slider)
- `obs_keys` keys selectable in the observable menu 
- `print_parameters = groups` parameter keys that get printed
- `autofit = false` fit SFS with sigmoid function and print intersection on <enter>
- `autoadvance = true` advance on <enter>
"""
function plot_GUI(
        df::DataFrames.DataFrame;
        parameters = filter(k -> k != :mc, collect(propertynames(df))),
        groups = find_groups(df, parameters),
        group_sizes = map(g -> length(unique(getproperty(df, g[1]))), groups),
        max_menu_size = 10,
        represent_as = Dict{Symbol, Symbol}(:beta => :ignore), # overwrite
        obs_keys = sort(collect(keys(df.mc[1])), by = string),
        print_parameters = vcat(collect.(groups[group_sizes .> 1])...),
        autofit = false, autoadvance_last = true,
        SFS_keys = (:SFD, :SFS, :SFW, :SFDx, :SFSx, :SFWx, :SFDy, :SFSy, :SFWy)
    )
    
    Ncontrols = length(groups) + 1
    representations = map(groups, group_sizes) do group, N
        rep = :unknown
        for name in group
            if haskey(represent_as, name)
                suggested = represent_as[name]
                if suggested in (:menu, :slider, :ignore)
                    if rep == :unknown
                        rep = suggested
                    elseif rep != suggested
                        @warn "$name should be represented as $suggested, but is already mapped to $rep as part of $group. Falling back to size based mapping."
                        rep = :fallback
                        break
                    end
                else
                    @warn "Representation $suggested for $name is invalid. Use :menu, :slider or :ignore. Ignoring"
                end
            end
        end
        
        if rep in (:unknown, :fallback)
            if N > max_menu_size; rep = :slider
            elseif N > 1;         rep = :menu
            else                  rep = :ignore
            end
        end
        
        return rep  
    end
    
    slider_groups = groups[representations .== :slider]
    menu_groups = groups[representations .== :menu]
    
    
    fig = Figure()
    
    controls = fig[1, 1] = GridLayout()
    
    menu_trigger = Makie.Observable(nothing)
    slider_trigger = Makie.Observable(nothing)
    
    # Generate Menus
    menus = []
    for (i, group) in enumerate(menu_groups)
        vals = map(key -> unique(getproperty(df, key)), group)
        idxs = sortperm(vals[1])
        
        menu_vals = map(idxs) do idx
            [Pair(group[i], vals[i][idx]) for i in eachindex(group)]
        end
        menu_labels = map(idxs) do idx
            mapreduce(param_string, (a, b) -> "$a, $b", getindex.(vals, idx), group)
        end
        
        x, y = fldmod1(i, max(2, length(slider_groups)))
        menu = Menu(controls[y, x], options = zip(menu_labels, menu_vals), width = 200)
        menu.i_selected[] = 1
        push!(menus, menu)
        on(_ -> notify(menu_trigger), menu.selection)
    end
    
    x, y = fldmod1(length(menu_groups) + 1, max(2, length(slider_groups)))
    obs_menu = Menu(controls[y, x], options = zip(string.(obs_keys), obs_keys), width = 200)
    obs_menu.i_selected[] = 1
    
    prefilter = Makie.Observable(BitVector(fill(true, length(df.mc))))
    on(menu_trigger) do _
        prefilter.val .= true
        for menu in menus
            if menu.selection[] isa Vector
                for (name, val) in menu.selection[]
                    prefilter.val = prefilter.val .& (getproperty(df, name) .== val)
                end
            else
                @warn "Empty menu"
            end
        end
        notify(prefilter)
        return
    end
    
    # Generate Sliders
    x += 1
    sliders = []
    slider_maps = []
    for (i, group) in enumerate(slider_groups)
        vals = map(prefilter) do prefilter
            map(key -> unique(getproperty(df[prefilter, :], key)), group)
        end
        idxs = map(vals -> sortperm(vals[1]), vals)
        
        sl = Slider(controls[i, x], range = idxs)
        on(_ -> notify(slider_trigger), sl.value)
        push!(sliders, sl)
        push!(slider_maps, vals)
        
        label = map(sl.value, vals) do idx, vals
            mapreduce(i -> param_string(vals[i][idx], group[i]), (a, b) -> "$a, $b", eachindex(group))
        end
        Label(controls[i, x+1], label)
    end
    
    # Generate inverters
    x += 2
    invert_x = Toggle(controls[1, x]).active
    invert_y = Toggle(controls[2, x]).active
    Label(controls[1, x+1], map(b -> ifelse(b, "1/x", "x"), invert_x))
    Label(controls[2, x+1], map(b -> ifelse(b, "1/y", "y"), invert_y))
    
    # DataFrame filter
    # Could probably make this a bit more efficient by caching filters (i.e. do individual on's early on)
    selection = BitVector(fill(false, length(df.mc)))
    mcs = map(prefilter, slider_trigger) do prefilter, _
        copyto!(selection, prefilter)
        for i in eachindex(sliders)
            _map = slider_maps[i][]
            idx = sliders[i].value[]
            for j in eachindex(slider_groups[i])
                name = slider_groups[i][j] 
                val = _map[j][idx]
                selection = selection .& (getproperty(df, name) .== val)
            end
        end
        
        sort(df[selection, :mc], by = mc -> mc.parameters.beta)
    end
    
    # Generate points
    ps = map(mcs, invert_x, invert_y, obs_menu.selection) do mcs, invert_x, invert_y, obs
        betas = map(mc -> mc.parameters.beta, mcs)
        xs = invert_x ? 1.0 ./ betas : betas
        data = map(mc -> mc[obs] |> mean |> real, mcs)
        ys = invert_y ? 1.0 ./ data : data
        Point2f.(xs, ys)
    end
    
    ax = Axis(
        fig[2, 1],
        xlabel = map(x -> x ? "Temperature T" : "Inverse Temperature β", invert_x)
    )
    
    # Limit reset
    onany(invert_x, invert_y, obs_menu.selection) do invert_x, invert_y, obs
        if obs in (:occ, :occs, :Mx, :My, :Mz) && !invert_y
            autolimits!(ax)
            ylims!(ax, 0, 1)
        else
            autolimits!(ax)
        end
    end
    
    scatterlines!(ax, ps)
    
    # SFD line
    ps2 = map(invert_x, invert_y) do invert_x, invert_y
        xs = invert_x ? range(1e-6, 1.0, length=50) : (1:30)
        ys = invert_x ? 2/pi .* xs : 2/pi ./ xs
        Point2f.(xs, invert_y ? 1.0 ./ ys : ys)
    end

    lines!(
        ax, ps2, 
        visible = map(key -> key in SFS_keys, obs_menu.selection),
        xautolimits = false, yautolimits = false
    )
    
    # Other interactivity
    println("Parameters printed: $print_parameters [β or T] [obs or 1 / obs]")
    on(events(fig.scene).mousebutton, priority = 10) do event
        if ispressed(fig, Keyboard.left_shift) && event.action == Mouse.press
            mc = first(mcs[])
            param = MonteCarlo.parameters(mc)
            param_str = mapreduce(key -> string(param[key]), (a, b) -> "$a $b", print_parameters)
            
            if event.button == Mouse.left
                x, y = mouseposition(ax.scene)
                println("$param_str $x $y; ")
                flush(stdout)
            elseif event.button == Mouse.right
                println("$param_str NaN NaN; ")
                flush(stdout)
            else
                return Consume(false)
            end

            if autoadvance_last
                sl = sliders[end]
                sl.selected_index[] += sl.selected_index[] != length(sl.range[])
            end
            return Consume()
        end
    end

    # Automatic T_C estimates for SFS
    if autofit
        # [max, 1.0, x @ half height, offset]
        f(xs, ps) = sigmoid(xs, ps[1], ps[2], ps[3], ps[4])

        ps = Makie.Observable(Point2f[(-1, -1), (-1, -1)])
        intersections = Makie.Observable([Point2f(NaN), Point2f(NaN)])

        onany(mcs, obs_menu.selection, invert_x, invert_y) do mcs, obs, invx, invy
            if obs in SFS_keys
                xs = map(mc -> 1.0 / mc.parameters.beta, mcs)
                ys = map(mc -> mean(mc[obs]) |> real, mcs)
                xs = xs[.!isnan.(ys)]
                ys = ys[.!isnan.(ys)]

                # Fit sigmoid
                try
                    _max = reduce((y1, y2) -> abs(y1) > abs(y2) ? y1 : y2, ys, init = 0.0)
                    half = reduce(eachindex(ys), init = 1) do i1, i2
                        abs(ys[i1] - 0.5 * _max) < abs(ys[i2] - 0.5 * _max) ? i1 : i2
                    end
                    fit = curve_fit(f, xs, ys, [_max, 1.0, xs[half], 0.0])

                    # show sigmoid
                    _xs = range(0.1, 1, length=100).^2
                    _ys = f(_xs, fit.param)
                    ps[] = Point2f.(invx ? _xs : 1.0 ./ _xs, invy ? 1 ./ _ys : _ys)
                

                    # find intersection (direct)
                    x = NaN
                    y = NaN
                    for i in 2:length(xs)
                        if (ys[i-1] < 2xs[i-1] / pi) && (ys[i] > 2xs[i] / pi)
                            a = (ys[i] - ys[i-1]) / (xs[i] - xs[i-1])
                            b = ys[i] - a * xs[i]
                            x = b / (2/pi - a)
                            y = a * x + b
                            if y <= 0.0
                                x = y = NaN
                            else
                                break
                            end
                        end
                    end
                    int1 = Point2f(invx ? x : 1/x, invy ? 1/y : y)

                    # find intersection from fit
                    x = NaN
                    y = NaN
                    for i in 2:length(_xs)
                        if (_ys[i-1] > 2_xs[i-1] / pi) && (_ys[i] < 2_xs[i] / pi)
                            a = (_ys[i] - _ys[i-1]) / (_xs[i] - _xs[i-1])
                            b = _ys[i] - a * _xs[i]
                            x = b / (2/pi - a)
                            y = a * x + b
                            if y <= 0.0
                                x = y = NaN
                            else
                                break
                            end
                        end
                    end
                    int2 = Point2f(invx ? x : 1/x, invy ? 1/y : y)

                    intersections[] = [int1, int2]
                catch e
                    @warn "Failed to create fit: " exception = e
                    ps[] = [Point2f(NaN), Point2f(NaN)]
                    intersections[] = [Point2f(NaN), Point2f(NaN)]
                end
            end
        end

        p = lines!(
            ax, ps, visible = map(k -> k in SFS_keys, obs_menu.selection),
            color = :lightgrey, linestyle = :dot, linewidth = 3
        )
        translate!(p, 0, 0, -1)
        scatter!(
            ax, intersections, color = :transparent, markersize=10, 
            strokecolor = [:blue, :red], strokewidth = 2,
            visible = map(k -> k in SFS_keys, obs_menu.selection)
        )

        on(events(fig).keyboardbutton, priority = 10) do event
            if event.key == Keyboard.enter && event.action == Keyboard.release
                mc = first(mcs[])
                param = MonteCarlo.parameters(mc)
                param_str = mapreduce(key -> string(param[key]), (a, b) -> "$a $b", print_parameters)
                
                if ispressed(fig, Keyboard.left_shift)
                    x, y = intersections[][1]
                    println("$param_str $x $y; ")
                else
                    x, y = intersections[][2]
                    println("$param_str $x $y; ")
                end
                flush(stdout)

                if autoadvance_last
                    sl = sliders[end]
                    sl.selected_index[] += sl.selected_index[] != length(sl.range[])
                end

                return Consume()
            end
        end

    end

    # Trigger update chain once
    notify(menu_trigger)
    
    fig
end