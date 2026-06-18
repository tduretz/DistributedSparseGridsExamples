using DistributedSparseGrids, StaticArrays, Printf
import DistributedSparseGrids: AbstractCollocationPoint, AbstractHierarchicalCollocationPoint, AbstractHierarchicalSparseGrid, numlevels, coord, pt_idx, i_multi, level, scaling_weight, fval
using GLMakie, Colors, Printf, MathTeXEngine
import Colors: distinguishable_colors, RGB, N0f8, colormap
import DistributedSparseGrids: generate_next_level, distributed_init_weights_inplace_ops
Makie.update_theme!( fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))
Makie.inline!(true)
const Ma = 1e6*365*24*3600

include("./src/sparse_grid_visualisation.jl")

function scale(x, xmin, ymin, xmax, ymax)
    a = (ymax-ymin) / (xmax-xmin)
    b = ymax - a*xmax
    return a*x + b 
end

function stress(x, ID)
    # println("simulation call $(x)")
    t = LinRange(0, 0.01Ma, 100)
    ε̇ = 1e-14
    # Defining parameter range
    ηmin, ηmax = 1e17, 1e22
    Gmin, Gmax = 1e9, 1e11
    # Scale from [-1; 1] to physical range
    η = scale(x[1], -1.0, ηmin, 1.0, ηmax)
    G = scale(x[2], -1.0, Gmin, 1.0, Gmax)
    # Compute model
    return @. 2*η*ε̇ * (1 - exp(- G/η * t))
end

function sparse_grid(N::Int, pointprops; nlevel=6, RT=Vector{Float64}, CT=Float64)
    # define collocation point
    CPType = CollocationPoint{N,CT}
    # define hierarchical collocation point
    HCPType = HierarchicalCollocationPoint{N,CPType,RT}
    # init grid
    asg = init(AHSG{N,HCPType},pointprops)
    #set of all collocation points
    cpts = Set{HierarchicalCollocationPoint{N,CPType,RT}}(collect(asg))
    # fully refine grid nlevel-1 times
    for i = 1:nlevel-1
        union!(cpts,generate_next_level!(asg))
    end
    return asg
end

let 
    t = LinRange(0, 0.01Ma, 100)
    ηmin, ηmax = 1e17, 1e22
    Gmin, Gmax = 1e9, 1e11

    # one forward model evaluation (test - useless) 
    x = [-0.95, 1.0]
    τ = stress(x, " ")

    # define point properties 
    #	1->closed point set
    # 	2->open point set
    #	3->left-open point set
    #	4->right-open point set
    asg = sparse_grid(2, @SVector [1,1]; nlevel=6) # closed point set in x[1] and x[2] --- standard 
 
    # initialize weights
    init_weights!(asg, stress)
    
    # refinement
    for i = 1:2
        cpts = generate_next_level!(asg, 1e-2, 20)
        init_weights!(asg, collect(cpts), stress)
    end

    # integration
    τ_integrated = integrate(asg)
    
    # interpolation
    x̄ = zeros(2)*2.0 .- 0.5
    τ̄ = interpolate(asg, x̄)	
    τ_truth = stress(x̄, " ")

    # visualisation
    f = Figure()
    ax = Axis(f[1,1], title=L"$$Viscoelastic loading", xlabel=L"$t$ [s]", ylabel=L"$\tau$ [Pa]" )
    scatter!(ax, t[1:10:end], τ̄[1:10:end], label=L"$$surroguate (specific point)")
    lines!(ax, t, τ_truth, label=L"$$truth (specific point)")
    lines!(ax, t, τ_integrated/2/2, label=L"$$mean (parameter space)") # division by 4 (volume of the space 2*2)
    Legend(f[1,2], ax)
    xtickformat = vals -> [@sprintf("%1.2e", scale(val, -1.0, ηmin, 1.0, ηmax)) for val in vals]
    ytickformat = vals -> [@sprintf("%1.2e", scale(val, -1.0, Gmin, 1.0, Gmax)) for val in vals]
    ax = Axis(f[2,1:2], xtickformat=xtickformat, xlabel=L"$\eta$ [Pa.s]", ytickformat=ytickformat, ylabel=L"$G$ [Pa]", title=L"$$Dense grid")
    Makie.scatter!(ax, asg)
    # GLMakie.surface!(ax, asg, 20, x->x[end])
    display(f)

end
