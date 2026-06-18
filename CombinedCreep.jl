using DistributedSparseGrids, StaticArrays, Printf
import DistributedSparseGrids: AbstractCollocationPoint, AbstractHierarchicalCollocationPoint, AbstractHierarchicalSparseGrid, numlevels, coord, pt_idx, i_multi, level, scaling_weight, fval
using GLMakie, Colors, Printf
import Colors: distinguishable_colors, RGB, N0f8, colormap
import DistributedSparseGrids: generate_next_level, distributed_init_weights_inplace_ops
Makie.update_theme!( fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))
Makie.inline!(true)
const Ma = 1e6*365*24*3600

include("./src/sparse_grid_visualisation.jl")

function stress(x, ID)

    # Scale from [-1; 1] to physical range
    T    = scale(x[1], -1.0, 573, 1.0, 2073)
    logε̇ = scale(x[2], -1.0, -50, 1.0, 5)
    ε̇    = 10^logε̇

    # stress for diffusion
    ηdif = 1//2 * Cdif^(-1/ndif) * d^(mdif/ndif) * exp(Edif/ndif/R/T) * ε̇^(1/ndif-1)
    τdif = 2ηdif*ε̇

    # stress for dislocation
    ηdis = 1//2 * Cdis^(-1/ndis) * d^(mdis/ndis) * exp(Edis/ndis/R/T) * ε̇^(1/ndis-1)
    τdis = 2ηdis*ε̇

    return Float64[abs.(log10(τdif) .- log10(τdis)), log10(τdif), log10(τdis)]
end

function scale(x, xmin, ymin, xmax, ymax)
    a = (ymax-ymin) / (xmax-xmin)
    b = ymax - a*xmax
    return a*x + b 
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

const R    = 8.314 # gas constant
const d    = 2e-3  # grain size

# Data for dislocation creep of olivine
const Cdis = 1.1e-16
const ndis = 3.5
const mdis = 0.0
const Edis = 530e3
const Vdis = 11e-6

# Data for diffusion creep of olivine
const Cdif = 1.5e-15
const ndif = 1.0
const mdif = 3.0
const Edif = 375e3
const Vdif = 4e-6

#let
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
    #comparefct(x) = abs(x[1])>1e-4 || abs(x[2])>1e-2 || abs(x[3])>1e-1 # use all 3 fields as refinement indicator with different tolerances
    comparefct(x) = abs(x[1])>1e-4 # use only the first field as a refinement indicator
    for i = 1:15
        cpts = generate_next_level!(asg, comparefct, 20)
        init_weights!(asg, collect(cpts), stress)
    end

    # visualisation
    f = Figure()
    ax1 = Axis3(f[1,1], title=L"$$Stress difference")
    ax2 = Axis3(f[2,1], title=L"$\tau_\text{dif}$ and $\tau_\text{dis}$")
    ax3 = Axis3(f[2,2], title=L"$\tau_\text{dif} < \tau_\text{dis}$")
    GLMakie.surface!(ax1, asg, 100, x->x[1])
    GLMakie.surface!(ax2, asg, 100, x->x[2])
    GLMakie.surface!(ax2, asg, 100, x->x[3], colormap=:jet)
    GLMakie.surface!(ax3, asg, 100, x->x[2]<x[3])
    ax4 = Axis(f[1,2])
    Makie.scatter!(ax1, asg)
    Makie.scatter!(ax2, asg)
    #Makie.scatter!(ax3, asg)
    Makie.scatter!(ax4, asg)
    display(f)
#end