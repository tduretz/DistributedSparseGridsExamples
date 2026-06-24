function Makie.scatter!(ax::AxisType, sg::SG; markersize=10, z_offset=hcpt->0.0) where {AxisType<:Makie.AbstractAxis, CT,CP<:AbstractCollocationPoint{2,CT},HCP<:AbstractHierarchicalCollocationPoint{2,CP},SG<:AbstractHierarchicalSparseGrid{2,HCP}}
    colors = cols = map(x->RGBA{Float64}(x.r,x.g,x.b,1.0), distinguishable_colors(numlevels(sg)+1, [RGB(1,1,1)])[2:end])
    nlevel = numlevels(sg)
    #traces = Vector{GenericTrace}(undef,nlevel)
    xvals = Vector{Vector{CT}}(undef,nlevel)
    yvals = Vector{Vector{CT}}(undef,nlevel)
    zvals = Vector{Vector{CT}}(undef,nlevel)
    #text = Vector{Vector{String}}(undef,nlevel)
    clr = Vector{Vector{RGB{N0f8}}}(undef,nlevel)
    isaxis2d = AxisType <: Axis ? true : (AxisType <: Axis3 ? false : error())
    for l = 1:nlevel
                  xvals[l] = Vector{CT}()
                  yvals[l] = Vector{CT}()
                  zvals[l] = Vector{CT}()
                  clr[l] = Vector{RGB{N0f8}}()
                  #text[l] = Vector{String}()
    end
    for hcpt in sg
                  l = level(hcpt)
                  push!(xvals[l],coord(hcpt,1))
                  push!(yvals[l],coord(hcpt,2))
                  push!(zvals[l],z_offset(hcpt))
                  #push!(zvals,interpolate(sg, [xvals[end], yvals[end]]))
                  #push!(text[l],string(pt_idx(hcpt))*"^"*string(i_multi(hcpt)))
                  push!(clr[l],colors[level(hcpt)])
    end
    for i = 1:nlevel
                  mw = markersize-foldl((x,y)->x+2.0/(y),1:i)
                  if isaxis2d
                                Makie.scatter!(ax, xvals[i], yvals[i], markersize=mw, color=clr[i])
                  else
                                Makie.scatter!(ax, xvals[i], yvals[i], zvals[i], markersize=mw, color=clr[i])
                  end
                  #traces[i] = p = PlotlyJS.scatter(x=xvals[i], y=yvals[i], text=text[i], marker_color=clr[i], mode="markers",marker_size=mw,textposition="bottom center",name="level $i")
    end
    return nothing
end

function Makie.scatter!(ax::Axis3, sg::SG; markersize=10, z_offset=hcpt->0.0) where {CT,CP<:AbstractCollocationPoint{3,CT},HCP<:AbstractHierarchicalCollocationPoint{3,CP},SG<:AbstractHierarchicalSparseGrid{3,HCP}}
    colors = cols = map(x->RGBA{Float64}(x.r,x.g,x.b,1.0), distinguishable_colors(numlevels(sg)+1, [RGB(1,1,1)])[2:end])
    nlevel = numlevels(sg)
    #traces = Vector{GenericTrace}(undef,nlevel)
    xvals = Vector{Vector{CT}}(undef,nlevel)
    yvals = Vector{Vector{CT}}(undef,nlevel)
    zvals = Vector{Vector{CT}}(undef,nlevel)
    #text = Vector{Vector{String}}(undef,nlevel)
    clr = Vector{Vector{RGB{N0f8}}}(undef,nlevel)
    for l = 1:nlevel
        xvals[l] = Vector{CT}()
        yvals[l] = Vector{CT}()
        zvals[l] = Vector{CT}()
        clr[l] = Vector{RGB{N0f8}}()
        #text[l] = Vector{String}()
    end
    for hcpt in sg
        l = level(hcpt)
        push!(xvals[l],coord(hcpt,1))
        push!(yvals[l],coord(hcpt,2))
        push!(zvals[l],coord(hcpt,3))
        #push!(zvals,interpolate(sg, [xvals[end], yvals[end]]))
        #push!(text[l],string(pt_idx(hcpt))*"^"*string(i_multi(hcpt)))
        push!(clr[l],colors[level(hcpt)])
    end
    for i = 1:nlevel
        mw = markersize-foldl((x,y)->x+2.0/(y),1:i)
        Makie.scatter!(ax, xvals[i], yvals[i], zvals[i], markersize=mw, color=clr[i])
    end
    return nothing
end

function GLMakie.surface!(ax, asg::SG, npts = 20, postfun=x->x; kwargs...) where {CT,CP<:AbstractCollocationPoint{2,CT},HCP<:AbstractHierarchicalCollocationPoint{2,CP},SG<:AbstractHierarchicalSparseGrid{2,HCP}}
    xs = LinRange(-1.0, 1.0, npts)
    ys = LinRange(-1.0, 1.0, npts)
    rcp = first(asg)
    tmp = zero(scaling_weight(rcp))
    zs = [begin;
                                interpolate!(tmp, asg, [x, y])
                                postfun(tmp)
                  end
                  for x in xs, y in ys]
    return GLMakie.surface!(ax, xs, ys, zs; kwargs...)
end
function GLMakie.surface!(ax, npts = 20, postfun=x->x; kwargs...)
    xs = LinRange(-1.0, 1.0, npts)
    ys = LinRange(-1.0, 1.0, npts)
    zs = [begin;
            postfun([x, y])
        end
        for x in xs, y in ys]
    return GLMakie.surface!(ax, xs, ys, zs; kwargs...)
end